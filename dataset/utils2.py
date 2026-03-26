import re
import pandas as pd  # 新增：导入pandas用于处理Excel
from tqdm import tqdm
import torch
import torch.nn.functional as F


def pre_question(question, max_ques_words):
    question = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        question.lower(),
    ).replace('-', ' ').replace('/', ' ')
    question = question.rstrip(' ')

    # truncate question
    question_words = question.split(' ')
    if len(question_words) > max_ques_words:
        question = ' '.join(question_words[:max_ques_words])

    return question


def pre_caption(caption, max_words):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    return caption


from vqaTools.vqaEval import VQAEval
from refTools.refEvaluation import RefEvaluation

import json
import os
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

import utils


def vqa_eval(vqa, result_file, test_ques_path):
    vqaRes = vqa.loadRes(result_file, test_ques_path)
    # create vqaEval object by taking vqa and vqaRes
    vqaEval = VQAEval(vqa, vqaRes, n=2)  # n is precision of accuracy (number of places after decimal), default is 2
    # evaluate results
    vqaEval.evaluate()

    # print accuracies
    print("\n")
    print("Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
    print("Per Answer Type Accuracy is the following:")
    for ansType in vqaEval.accuracy['perAnswerType']:
        print("%s : %.02f" % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
    print("\n")

    return vqaEval


def collect_result(result, result_dir, filename, is_json=True, is_list=True):
    if is_json:
        result_file = os.path.join(result_dir, '%s_rank%d.json' % (filename, utils.get_rank()))
        final_result_file = os.path.join(result_dir, '%s.json' % filename)
        json.dump(result, open(result_file, 'w'))
    else:
        result_file = os.path.join(result_dir, '%s_rank%d.pth' % (filename, utils.get_rank()))
        final_result_file = os.path.join(result_dir, '%s.pth' % filename)
        torch.save(result, result_file)

    dist.barrier()

    result = None
    if utils.is_main_process():
        # combine results from all processes
        if is_list:
            result = []
        else:
            result = {}
        for rank in range(utils.get_world_size()):
            if is_json:
                result_file = os.path.join(result_dir, '%s_rank%d.json' % (filename, rank))
                res = json.load(open(result_file, 'r'))
            else:
                result_file = os.path.join(result_dir, '%s_rank%d.pth' % (filename, rank))
                res = torch.load(result_file)
            if is_list:
                result += res
            else:
                result.update(res)

    return result


def save_result(result, result_dir, filename, is_json=True, is_list=True):
    if is_json:
        result_file = os.path.join(result_dir, '%s_rank%d.json' % (filename, utils.get_rank()))
        final_result_file = os.path.join(result_dir, '%s.json' % filename)
        json.dump(result, open(result_file, 'w'))
    else:
        result_file = os.path.join(result_dir, '%s_rank%d.pth' % (filename, utils.get_rank()))
        final_result_file = os.path.join(result_dir, '%s.pth' % filename)
        torch.save(result, result_file)

    dist.barrier()

    if utils.is_main_process():
        # combine results from all processes
        if is_list:
            result = []
        else:
            result = {}
        for rank in range(utils.get_world_size()):
            if is_json:
                result_file = os.path.join(result_dir, '%s_rank%d.json' % (filename, rank))
                res = json.load(open(result_file, 'r'))
            else:
                result_file = os.path.join(result_dir, '%s_rank%d.pth' % (filename, rank))
                res = torch.load(result_file)
            if is_list:
                result += res
            else:
                result.update(res)
        if is_json:
            json.dump(result, open(final_result_file, 'w'))
        else:
            torch.save(result, final_result_file)

        print('result file saved to %s' % final_result_file)
    dist.barrier()
    return final_result_file


def grounding_eval(results, dets, cocos, refer, alpha, mask_size=24, save_excel_path='grounding_results.xlsx'):
    """
    计算grounding评估指标，并将详细结果保存到Excel

    Args:
        results: 模型预测结果列表
        dets: 检测框数据
        cocos: coco格式数据
        refer: REFER工具类实例
        alpha: IoU计算的权重参数
        mask_size: mask的尺寸
        save_excel_path: Excel文件保存路径

    Returns:
        eval_result: 各数据集的准确率字典
    """
    correct_A_d, correct_B_d, correct_val_d = 0, 0, 0
    correct_A, correct_B, correct_val = 0, 0, 0
    num_A, num_B, num_val = 0, 0, 0

    # 新增：用于保存Excel数据的列表
    excel_data = []

    for res in tqdm(results):
        ref_id = res['ref_id']
        ref = refer.Refs[ref_id]
        ref_box = refer.refToAnn[ref_id]['bbox']
        image = refer.Imgs[ref['image_id']]

        mask = res['pred'].cuda().view(1, 1, mask_size, mask_size)
        mask = F.interpolate(mask, size=(image['height'], image['width']), mode='bicubic').squeeze()

        # rank detection boxes
        max_score = 0
        pred_box = None
        for det in dets[str(ref['image_id'])]:
            score = mask[int(det[1]):int(det[1] + det[3]), int(det[0]):int(det[0] + det[2])]
            area = det[2] * det[3]
            score = score.sum() / area ** alpha
            if score > max_score:
                pred_box = det[:4]
                max_score = score

        IoU_det = computeIoU(ref_box, pred_box)
        # 新增：判断是否攻击成功（IoU≥0.5表示攻击失败，<0.5表示攻击成功）
        attack_success = 1 if IoU_det < 0.5 else 0
        attack_success_str = "成功" if attack_success else "失败"

        # 原有逻辑：统计各split的准确率
        if ref['split'] == 'testA':
            num_A += 1
            if IoU_det >= 0.5:
                correct_A_d += 1
        elif ref['split'] == 'testB':
            num_B += 1
            if IoU_det >= 0.5:
                correct_B_d += 1
        elif ref['split'] == 'val':
            num_val += 1
            if IoU_det >= 0.5:
                correct_val_d += 1

                # 新增：收集当前样本的详细信息
        excel_data.append({
            'ref_id': ref_id,
            'image_id': ref['image_id'],
            'split': ref['split'],
            'IoU': round(IoU_det, 4),
            'attack_success': attack_success,
            'attack_success_str': attack_success_str,
            'ref_box': ref_box,
            'pred_box': pred_box
        })

    # 新增：将数据保存到Excel
    df = pd.DataFrame(excel_data)
    # 创建多级统计（可选）
    df_summary = df.groupby('split').agg({
        'attack_success': ['count', 'sum', 'mean'],
        'IoU': ['mean', 'std']
    }).round(4)
    df_summary.columns = ['样本总数', '攻击成功数', '攻击成功率', '平均IoU', 'IoU标准差']

    # 保存到Excel（包含详细数据和统计摘要）
    with pd.ExcelWriter(save_excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='详细结果', index=False)
        df_summary.to_excel(writer, sheet_name='统计摘要')

    print(f"\n详细结果已保存到Excel文件：{save_excel_path}")

    # 计算最终评估结果
    eval_result = {
        'val_d': correct_val_d / num_val if num_val > 0 else 0,
        'testA_d': correct_A_d / num_A if num_A > 0 else 0,
        'testB_d': correct_B_d / num_B if num_B > 0 else 0
    }

    for metric, acc in eval_result.items():
        print(f'{metric}: {acc:.3f}')

    return eval_result


# IoU function
def computeIoU(box1, box2):
    # each box is of [x1, y1, w, h]
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[0] + box1[2] - 1, box2[0] + box2[2] - 1)
    inter_y2 = min(box1[1] + box1[3] - 1, box2[1] + box2[3] - 1)

    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
    else:
        inter = 0
    union = box1[2] * box1[3] + box2[2] * box2[3] - inter
    return float(inter) / union if union > 0 else 0