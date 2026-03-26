import re
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd

def pre_question(question,max_ques_words):
    question = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        question.lower(),
    ).replace('-', ' ').replace('/', ' ')  
    question = question.rstrip(' ')
    
    #truncate question
    question_words = question.split(' ')
    if len(question_words)>max_ques_words:
        question = ' '.join(question_words[:max_ques_words])
            
    return question


def pre_caption(caption,max_words):
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

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
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
from tqdm import tqdm


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
        result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,utils.get_rank()))
        final_result_file = os.path.join(result_dir, '%s.json'%filename)
        json.dump(result,open(result_file,'w'))
    else:
        result_file = os.path.join(result_dir, '%s_rank%d.pth'%(filename,utils.get_rank()))
        final_result_file = os.path.join(result_dir, '%s.pth'%filename)
        torch.save(result,result_file)     
        
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
                result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,rank))
                res = json.load(open(result_file,'r'))
            else:
                result_file = os.path.join(result_dir, '%s_rank%d.pth'%(filename,rank))
                res = torch.load(result_file)            
            if is_list:
                result += res
            else:
                result.update(res) 
      
    return result    

    
def save_result(result, result_dir, filename, is_json=True, is_list=True):
    if is_json:
        result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,utils.get_rank()))
        final_result_file = os.path.join(result_dir, '%s.json'%filename)
        json.dump(result,open(result_file,'w'))
    else:
        result_file = os.path.join(result_dir, '%s_rank%d.pth'%(filename,utils.get_rank()))
        final_result_file = os.path.join(result_dir, '%s.pth'%filename)
        torch.save(result,result_file)     
        
    dist.barrier()

    if utils.is_main_process():   
        # combine results from all processes
        if is_list:
            result = []
        else:
            result = {}
        for rank in range(utils.get_world_size()):
            if is_json:
                result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,rank))
                res = json.load(open(result_file,'r'))
            else:
                result_file = os.path.join(result_dir, '%s_rank%d.pth'%(filename,rank))
                res = torch.load(result_file)            
            if is_list:
                result += res
            else:
                result.update(res)
        if is_json:                  
            json.dump(result,open(final_result_file,'w'))   
        else:            
            torch.save(result,final_result_file)     
        
        print('result file saved to %s'%final_result_file)
    dist.barrier()        
    return final_result_file

#
#
# def grounding_eval(results,dets,cocos,refer,alpha,mask_size=24):
#
#     correct_A_d, correct_B_d, correct_val_d = 0, 0, 0
#     correct_A, correct_B, correct_val = 0, 0, 0
#     num_A,num_B,num_val = 0,0,0
#
#     for res in tqdm(results):
#
#         ref_id = res['ref_id']
#         ref = refer.Refs[ref_id]
#         ref_box = refer.refToAnn[ref_id]['bbox']
#         image = refer.Imgs[ref['image_id']]
#
#         mask = res['pred'].cuda().view(1,1,mask_size,mask_size)
#         mask = F.interpolate(mask,size = (image['height'],image['width']), mode='bicubic').squeeze()
#
#         # rank detection boxes
#         max_score = 0
#         for det in dets[str(ref['image_id'])]:
#             score = mask[int(det[1]):int(det[1]+det[3]),int(det[0]):int(det[0]+det[2])]
#             area = det[2]*det[3]
#             score = score.sum() / area**alpha
#             if score>max_score:
#                 pred_box = det[:4]
#                 max_score = score
#
#         IoU_det = computeIoU(ref_box, pred_box)
#
#         if ref['split']=='testA':
#             num_A += 1
#             if IoU_det >= 0.5:
#                 correct_A_d += 1
#         elif ref['split']=='testB':
#             num_B += 1
#             if IoU_det >= 0.5:
#                 correct_B_d += 1
#         elif ref['split']=='val':
#             num_val += 1
#             if IoU_det >= 0.5:
#                 correct_val_d += 1
#
#     eval_result = {'val_d':correct_val_d/num_val,'testA_d':correct_A_d/num_A,'testB_d':correct_B_d/num_B}
#
#     for metric, acc in eval_result.items():
#         print(f'{metric}: {acc:.3f}')
#
#     return eval_result


def grounding_eval(results, dets, cocos, refer, alpha, mask_size=24):
    correct_A_d, correct_B_d, correct_val_d = 0, 0, 0
    correct_A, correct_B, correct_val = 0, 0, 0
    num_A, num_B, num_val = 0, 0, 0

    # 初始化存储不同split结果的列表
    testA_data = []
    testB_data = []
    val_data = []
    # 初始化各split的样本序号
    testA_idx = 1
    testB_idx = 1
    val_idx = 1

    for res in tqdm(results):

        ref_id = res['ref_id']
        ref = refer.Refs[ref_id]
        ref_box = refer.refToAnn[ref_id]['bbox']
        image = refer.Imgs[ref['image_id']]

        mask = res['pred'].cuda().view(1, 1, mask_size, mask_size)
        mask = F.interpolate(mask, size=(image['height'], image['width']), mode='bicubic').squeeze()

        # rank detection boxes
        max_score = 0
        pred_box = None  # 初始化pred_box，避免未找到时报错
        for det in dets[str(ref['image_id'])]:
            score = mask[int(det[1]):int(det[1] + det[3]), int(det[0]):int(det[0] + det[2])]
            area = det[2] * det[3]
            score = score.sum() / area ** alpha
            if score > max_score:
                pred_box = det[:4]
                max_score = score

                # 增加pred_box为空的异常处理
        if pred_box is None:
            IoU_det = 0.0
        else:
            IoU_det = computeIoU(ref_box, pred_box)

        # 判断IoU是否达标，转换为1/0
        is_correct = 1 if IoU_det >= 0.5 else 0

        # 按split分类存储结果，并更新计数
        if ref['split'] == 'testA':
            num_A += 1
            if IoU_det >= 0.5:
                correct_A_d += 1
            # 将当前样本结果添加到testA列表
            testA_data.append({'样本序号': testA_idx, 'IoU是否达标(≥0.5)': is_correct})
            testA_idx += 1
        elif ref['split'] == 'testB':
            num_B += 1
            if IoU_det >= 0.5:
                correct_B_d += 1
            # 将当前样本结果添加到testB列表
            testB_data.append({'样本序号': testB_idx, 'IoU是否达标(≥0.5)': is_correct})
            testB_idx += 1
        elif ref['split'] == 'val':
            num_val += 1
            if IoU_det >= 0.5:
                correct_val_d += 1
            # 将当前样本结果添加到val列表
            val_data.append({'样本序号': val_idx, 'IoU是否达标(≥0.5)': is_correct})
            val_idx += 1

            # 将结果保存到Excel，不同split对应不同sheet
    with pd.ExcelWriter('VGexcel/grounding_eval_results_coattackmda.xlsx', engine='openpyxl') as writer:
        # 保存testA结果
        if testA_data:
            pd.DataFrame(testA_data).to_excel(writer, sheet_name='testA', index=False)
        # 保存testB结果
        if testB_data:
            pd.DataFrame(testB_data).to_excel(writer, sheet_name='testB', index=False)
        # 保存val结果
        if val_data:
            pd.DataFrame(val_data).to_excel(writer, sheet_name='val', index=False)

    # 计算评估指标
    eval_result = {
        'val_d': correct_val_d / num_val if num_val > 0 else 0.0,
        'testA_d': correct_A_d / num_A if num_A > 0 else 0.0,
        'testB_d': correct_B_d / num_B if num_B > 0 else 0.0
    }

    # 打印评估结果
    for metric, acc in eval_result.items():
        print(f'{metric}: {acc:.3f}')

    return eval_result


# IoU function
def computeIoU(box1, box2):
    # each box is of [x1, y1, w, h]
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[0]+box1[2]-1, box2[0]+box2[2]-1)
    inter_y2 = min(box1[1]+box1[3]-1, box2[1]+box2[3]-1)

    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
    else:
        inter = 0
    union = box1[2]*box1[3] + box2[2]*box2[3] - inter
    return float(inter)/union
        
        
        