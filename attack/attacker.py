import numpy as np
import torch
import torch.nn as nn

import copy
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
# from nltk.corpus import wordnet2021
from gensim.models import KeyedVectors

glovemodel = KeyedVectors.load('attack/glove2word2vev300d.model')


filter_words = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
                'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
                'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
                'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
                'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
                "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
                'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
                'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
                'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
                'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
                'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
                "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
                'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
                'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
                'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
                'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
                'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'The','the', 'their', 'theirs',
                'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
                'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
                'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
                'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
                'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
                "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
                'your', 'yours', 'yourself', 'yourselves', '.', '-', 'a the', '/', '?', 'some', '"', ',', 'b', '&', '!',
                '@', '%', '^', '*', '(', ')', "-", '-', '+', '=', '<', '>', '|', ':', ";", '～', '·','. .','~']
filter_words = set(filter_words)


class TextAttacker():
    def __init__(self, ref_net, tokenizer, cls=True, max_length=30, number_perturbation=1, topk=50,
                 threshold_pred_score=0.3, batch_size=32):
        self.ref_net = ref_net
        self.tokenizer = tokenizer
        self.max_length = max_length
        # epsilon_txt
        self.num_perturbation = number_perturbation
        self.threshold_pred_score = threshold_pred_score
        self.topk = topk
        self.batch_size = batch_size
        self.cls = cls

    def img_guided_attack(self, net, texts, img_embeds=None):
        device = self.ref_net.device

        text_inputs = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_length,
                                     return_tensors='pt').to(device)

        # substitutes
        mlm_logits = self.ref_net(text_inputs.input_ids, attention_mask=text_inputs.attention_mask).logits

        word_pred_scores_all, word_predictions = torch.topk(mlm_logits, self.topk, -1)  # seq-len k

        # original state
        origin_output = net.inference_text(text_inputs)
        if self.cls:
            origin_embeds = origin_output['text_feat'][:, 0, :].detach()
        else:
            origin_embeds = origin_output['text_feat'].flatten(1).detach()

        final_adverse = []
        for i, text in enumerate(texts):
            # word importance eval
            # important_scores = self.get_important_scores(text, net, origin_embeds[i], self.batch_size, self.max_length)
            important_scores = self.get_important_scores2(text, net, img_embeds, self.batch_size, self.max_length)
            list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)

            words, sub_words, keys = self._tokenize(text)
            final_words = copy.deepcopy(words)
            change = 0

            for top_index in list_of_index:
                if change >= self.num_perturbation:
                    break
                tgt_word = words[top_index[0]]

                if tgt_word in filter_words:
                    continue
                if keys[top_index[0]][0] > self.max_length - 2:
                    continue
                #
                # substitutes = word_predictions[i, keys[top_index[0]][0]:keys[top_index[0]][1]]  # L, k
                # word_pred_scores = word_pred_scores_all[i, keys[top_index[0]][0]:keys[top_index[0]][1]]
                #
                # substitutes = get_substitues(substitutes, self.tokenizer, self.ref_net, 1, word_pred_scores,
                #                              self.threshold_pred_score)



                # Embedding Guidance
                substitutes =[]
                # try:
                # tgt_word = tgt_word.lower()

                def get_similar_words(glovemodel, word, topn=10):
                    """安全获取相似词的函数"""
                    # 尝试原词
                    if word in glovemodel.key_to_index:
                        return glovemodel.most_similar(word, topn=topn)

                    # 尝试小写
                    word_lower = word.lower()
                    if word_lower in glovemodel.key_to_index:
                        return glovemodel.most_similar(word_lower, topn=topn)

                    # 尝试去除标点/空格
                    word_clean = word.strip().strip(':,;?<>/\\|~@#$%^&*_-+=.')
                    if word_clean in glovemodel.key_to_index:
                        return glovemodel.most_similar(word_clean, topn=topn)

                    # 最后返回空列表或默认值
                    print(f"单词 '{word}' 及其变体均不在词汇表中")
                    return []

                # 使用这个函数替代原有的调用
                similar_words = get_similar_words(glovemodel, tgt_word)
                # similar_words = glovemodel.most_similar(tgt_word)
                substitutes = [word for word, _ in similar_words]
                # except KeyError:
                #
                #     mlm_logits = self.ref_net(text_inputs.input_ids,
                #                               attention_mask=text_inputs.attention_mask).logits
                #     word_pred_scores_all, word_predictions = torch.topk(mlm_logits, self.topk, -1)  # seq-len k
                #     substitutes = word_predictions[i, keys[top_index[0]][0]:keys[top_index[0]][1]]  # L, k
                #
                #     word_pred_scores = word_pred_scores_all[i, keys[top_index[0]][0]:keys[top_index[0]][1]]
                #
                #     substitutes = get_substitues(substitutes, self.tokenizer, self.ref_net, 1, word_pred_scores,
                #                                  self.threshold_pred_score)
                #     pass

                # print(substitute)
                replace_texts = [' '.join(final_words)]
                available_substitutes = [tgt_word]
                for substitute_ in substitutes:
                    substitute = substitute_

                    if substitute == tgt_word:
                        continue  # filter out original word
                    if '##' in substitute:
                        continue  # filter out sub-word

                    if substitute in filter_words:
                        continue
                    '''
                    # filter out atonyms
                    if substitute in w2i and tgt_word in w2i:
                        if cos_mat[w2i[substitute]][w2i[tgt_word]] < 0.4:
                            continue
                    '''
                    temp_replace = copy.deepcopy(final_words)
                    temp_replace[top_index[0]] = substitute
                    available_substitutes.append(substitute)
                    replace_texts.append(' '.join(temp_replace))
                replace_text_input = self.tokenizer(replace_texts, padding='max_length', truncation=True,
                                                    max_length=self.max_length, return_tensors='pt').to(device)
                replace_output = net.inference_text(replace_text_input)
                if self.cls:
                    replace_embeds = replace_output['text_feat'][:, 0, :]
                else:
                    replace_embeds = replace_output['text_feat'].flatten(1)

                loss = self.loss_func(replace_embeds, img_embeds, i)
                candidate_idx = loss.argmax()

                final_words[top_index[0]] = available_substitutes[candidate_idx]

                if available_substitutes[candidate_idx] != tgt_word:
                    change += 1

            final_adverse.append(' '.join(final_words))

        return final_adverse

    def img_guided_attack2(self, net, texts, img_embeds=None):
        device = self.ref_net.device

        text_inputs = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        ).to(device)

        # MLM logits (for fallback)
        mlm_logits = self.ref_net(
            text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask
        ).logits

        word_pred_scores_all, word_predictions = torch.topk(
            mlm_logits, self.topk, -1
        )  # seq_len x k

        final_adverse = []

        for i, text in enumerate(texts):

            words, sub_words, keys = self._tokenize(text)

            print("\n" + "=" * 80)
            print(f"[Original Text {i}]: {text}")
            print("=" * 80 + "\n")

            # 对每一个单词都寻找候选词
            for idx, tgt_word in enumerate(words):

                # basic filters
                if tgt_word in filter_words:
                    continue
                if keys[idx][0] > self.max_length - 2:
                    continue

                substitutes = []
                source = ""

                # ---------- GloVe candidates ----------
                # try:
                #     similar_words = glovemodel.most_similar(tgt_word)
                #     substitutes = [word for word, _ in similar_words]
                #     source = "GloVe"

                # ---------- fallback to BERT-MLM ----------
                # except KeyError:
                substitutes = word_predictions[
                              i, keys[idx][0]:keys[idx][1]
                              ]  # L x k

                word_pred_scores = word_pred_scores_all[
                                   i, keys[idx][0]:keys[idx][1]
                                   ]

                substitutes = get_substitues(
                    substitutes,
                    self.tokenizer,
                    self.ref_net,
                    1,
                    word_pred_scores,
                    self.threshold_pred_score
                )
                #     mlm_logits = self.ref_net(text_inputs.input_ids,
                #                               attention_mask=text_inputs.attention_mask).logits
                #     word_pred_scores_all, word_predictions = torch.topk(mlm_logits, self.topk, -1)  # seq-len k
                #     substitutes = word_predictions[i, keys[top_index[0]][0]:keys[top_index[0]][1]]  # L, k
                #
                #     word_pred_scores = word_pred_scores_all[i, keys[top_index[0]][0]:keys[top_index[0]][1]]
                #
                #     substitutes = get_substitues(substitutes, self.tokenizer, self.ref_net, 1, word_pred_scores,
                #                                  self.threshold_pred_score)



                source = "BERT-MLM"

                # ---------- clean candidates ----------
                clean_substitutes = []
                for sub in substitutes:
                    if sub == tgt_word:
                        continue
                    if '##' in sub:
                        continue
                    # if sub in filter_words:
                    #     continue
                    clean_substitutes.append(sub)

                # ---------- print ----------
                print(f"[Word {idx}] {tgt_word}")
                print(f"  Source     : {source}")
                print(f"  #Candidates: {len(clean_substitutes)}")
                print(f"  Candidates : {clean_substitutes}\n")

            # 这里不做攻击，仅占位保持接口一致
            final_adverse.append(text)

        return final_adverse

    def img_guided_attack3(self, net, texts, img_embeds=None):
        device = self.ref_net.device
        final_adverse = []

        for i, text in enumerate(texts):

            words, sub_words, keys = self._tokenize(text)

            print("\n" + "=" * 80)
            print(f"[Original Text {i}]: {text}")
            print("=" * 80 + "\n")

            # 对每一个单词寻找候选词
            for idx, tgt_word in enumerate(words):

                # ---------- basic filters ----------
                if tgt_word in filter_words:
                    continue
                if keys[idx][0] > self.max_length - 1:
                    continue

                # ---------- 构造 mask 后的 input_ids ----------
                masked_input_ids = self.tokenizer.encode(
                    text,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length
                )

                start, end = keys[idx]



                # 用 [MASK] 替换该词对应的所有 subword
                for pos in range(start, end):
                    masked_input_ids[pos] = self.tokenizer.mask_token_id

                masked_input_ids = torch.tensor(
                    masked_input_ids, device=device
                ).unsqueeze(0)

                attention_mask = (masked_input_ids != self.tokenizer.pad_token_id).long()

                # ---------- MLM forward ----------
                with torch.no_grad():
                    mlm_logits = self.ref_net(
                        masked_input_ids,
                        attention_mask=attention_mask
                    ).logits
                # 当前词仍然 mask


                # 取后一个词的位置作为候选
                # if idx + 1 < len(keys):
                #     # continue  # 最后一个词没后继，跳过
                #     cand_start, cand_end = keys[idx +1]
                # else:
                cand_start, cand_end = start, end  # 用自己
                # 用“后一个词”的 logits 取 top-k
                mask_logits = mlm_logits[0, cand_start:cand_end]


                # ---------- 取 mask 位置的 top-k ----------
                # mask_logits = mlm_logits[0, start:end]  # L × vocab
                word_pred_scores, word_predictions = torch.topk(
                    mask_logits, self.topk, dim=-1
                )  # L × k

                # ---------- 生成候选词 ----------
                substitutes = get_substitues(
                    word_predictions,
                    self.tokenizer,
                    self.ref_net,
                    1,
                    word_pred_scores,
                    self.threshold_pred_score
                )

                source = "BERT-MLM (masked)"

                # ---------- clean candidates ----------
                clean_substitutes = []
                for sub in substitutes:
                    if sub == tgt_word:
                        continue
                    if '##' in sub:
                        continue
                    clean_substitutes.append(sub)

                # ---------- print ----------
                print(f"[Word {idx}] {tgt_word}")
                print(f"  Source     : {source}")
                print(f"  #Candidates: {len(clean_substitutes)}")
                print(f"  Candidates : {clean_substitutes}\n")

            # 占位，保持接口一致
            final_adverse.append(text)

        return final_adverse
    def loss_func(self, txt_embeds, img_embeds, label):
        loss_TaIcpos = -txt_embeds.mul(img_embeds[label].repeat(len(txt_embeds), 1)).sum(-1)
        loss = loss_TaIcpos
        return loss

    def attack(self, net, texts):
        device = self.ref_net.device

        text_inputs = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_length,
                                     return_tensors='pt').to(device)

        # substitutes
        mlm_logits = self.ref_net(text_inputs.input_ids, attention_mask=text_inputs.attention_mask).logits
        word_pred_scores_all, word_predictions = torch.topk(mlm_logits, self.topk, -1)  # seq-len k

        # original state
        origin_output = net.inference_text(text_inputs)
        if self.cls:
            origin_embeds = origin_output['text_embed'][:, 0, :].detach()
        else:
            origin_embeds = origin_output['text_embed'].flatten(1).detach()

        criterion = torch.nn.KLDivLoss(reduction='none')
        final_adverse = []
        for i, text in enumerate(texts):
            # word importance eval
            important_scores = self.get_important_scores(text, net, origin_embeds[i], self.batch_size, self.max_length)

            list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)

            words, sub_words, keys = self._tokenize(text)
            final_words = copy.deepcopy(words)
            change = 0

            for top_index in list_of_index:
                if change >= self.num_perturbation:
                    break

                tgt_word = words[top_index[0]]
                if tgt_word in filter_words:
                    continue
                if keys[top_index[0]][0] > self.max_length - 2:
                    continue

                substitutes = word_predictions[i, keys[top_index[0]][0]:keys[top_index[0]][1]]  # L, k
                word_pred_scores = word_pred_scores_all[i, keys[top_index[0]][0]:keys[top_index[0]][1]]

                substitutes = get_substitues(substitutes, self.tokenizer, self.ref_net, 1, word_pred_scores,
                                             self.threshold_pred_score)

                replace_texts = [' '.join(final_words)]
                available_substitutes = [tgt_word]
                for substitute_ in substitutes:
                    substitute = substitute_

                    if substitute == tgt_word:
                        continue  # filter out original word
                    if '##' in substitute:
                        continue  # filter out sub-word

                    if substitute in filter_words:
                        continue
                    '''
                    # filter out atonyms
                    if substitute in w2i and tgt_word in w2i:
                        if cos_mat[w2i[substitute]][w2i[tgt_word]] < 0.4:
                            continue
                    '''
                    temp_replace = copy.deepcopy(final_words)
                    temp_replace[top_index[0]] = substitute
                    available_substitutes.append(substitute)
                    replace_texts.append(' '.join(temp_replace))
                replace_text_input = self.tokenizer(replace_texts, padding='max_length', truncation=True,
                                                    max_length=self.max_length, return_tensors='pt').to(device)
                replace_output = net.inference_text(replace_text_input)
                if self.cls:
                    replace_embeds = replace_output['text_embed'][:, 0, :]
                else:
                    replace_embeds = replace_output['text_embed'].flatten(1)

                loss = criterion(replace_embeds.log_softmax(dim=-1),
                                 origin_embeds[i].softmax(dim=-1).repeat(len(replace_embeds), 1))

                loss = loss.sum(dim=-1)
                candidate_idx = loss.argmax()

                final_words[top_index[0]] = available_substitutes[candidate_idx]

                if available_substitutes[candidate_idx] != tgt_word:
                    change += 1

            final_adverse.append(' '.join(final_words))

        return final_adverse

    def _tokenize(self, text):
        words = text.split(' ')

        sub_words = []
        keys = []
        index = 0
        for word in words:
            sub = self.tokenizer.tokenize(word)
            sub_words += sub
            keys.append([index, index + len(sub)])
            index += len(sub)

        return words, sub_words, keys

    def _get_masked(self, text):
        words = text.split(' ')
        len_text = len(words)
        masked_words = []
        for i in range(len_text):
            masked_words.append(words[0:i] + ['[UNK]'] + words[i + 1:])
        # list of words
        return masked_words

    def get_important_scores(self, text, net, origin_embeds, batch_size, max_length):
        device = origin_embeds.device

        masked_words = self._get_masked(text)
        masked_texts = [' '.join(words) for words in masked_words]  # list of text of masked words

        masked_embeds = []
        for i in range(0, len(masked_texts), batch_size):
            masked_text_input = self.tokenizer(masked_texts[i:i + batch_size], padding='max_length', truncation=True,
                                               max_length=max_length, return_tensors='pt').to(device)
            masked_output = net.inference_text(masked_text_input)
            if self.cls:
                masked_embed = masked_output['text_feat'][:, 0, :].detach()
            else:
                masked_embed = masked_output['text_feat'].flatten(1).detach()
            masked_embeds.append(masked_embed)
        masked_embeds = torch.cat(masked_embeds, dim=0)

        criterion = torch.nn.KLDivLoss(reduction='none')

        import_scores = criterion(masked_embeds.log_softmax(dim=-1),
                                  origin_embeds.softmax(dim=-1).repeat(len(masked_texts), 1))

        return import_scores.sum(dim=-1)

    def get_important_scores2(self, text, net, img_embed, batch_size, max_length):
            """
            Args:
                text: 原始文本（string）
                net: VLP 模型
                img_embed: 对应图像的 embedding，shape = [D]
                batch_size: batch size
                max_length: tokenizer max length
            Returns:
                import_scores: 每个 word 的重要性分数
            """

            device = img_embed.device

            # 1. 构造 masked texts
            masked_words = self._get_masked(text)
            masked_texts = [' '.join(words) for words in masked_words]

            masked_embeds = []
            for i in range(0, len(masked_texts), batch_size):
                masked_text_input = self.tokenizer(
                    masked_texts[i:i + batch_size],
                    padding='max_length',
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                ).to(device)

                masked_output = net.inference_text(masked_text_input)

                if self.cls:
                    masked_embed = masked_output['text_feat'][:, 0, :].detach()
                else:
                    masked_embed = masked_output['text_feat'].flatten(1).detach()

                masked_embeds.append(masked_embed)

            masked_embeds = torch.cat(masked_embeds, dim=0)  # [N_mask, D]

            # 2. Image-guided importance: KL( T_mask || I )
            # 将图像特征扩展到与 masked_embeds 对齐
            # img_embed = img_embed.unsqueeze(0).repeat(masked_embeds.size(0), 1)

            criterion = torch.nn.KLDivLoss(reduction='none')

            import_scores = criterion(
                masked_embeds.log_softmax(dim=-1),
                img_embed.softmax(dim=-1)
            )

            return import_scores.sum(dim=-1)

def get_substitues(substitutes, tokenizer, mlm_model, use_bpe, substitutes_score=None, threshold=3.0):
    # substitues L,k
    # from this matrix to recover a word
    words = []
    sub_len, k = substitutes.size()  # sub-len, k

    if sub_len == 0:
        return words

    elif sub_len == 1:
        for (i, j) in zip(substitutes[0], substitutes_score[0]):
            if threshold != 0 and j < threshold:
                break
            words.append(tokenizer._convert_id_to_token(int(i)))
    else:
        if use_bpe == 1:
            words = get_bpe_substitues(substitutes, tokenizer, mlm_model)
        else:
            return words
    #
    # print(words)
    return words


def get_bpe_substitues(substitutes, tokenizer, mlm_model):
    # substitutes L, k
    device = mlm_model.device
    substitutes = substitutes[0:12, 0:4]  # maximum BPE candidates

    # find all possible candidates

    all_substitutes = []
    for i in range(substitutes.size(0)):
        if len(all_substitutes) == 0:
            lev_i = substitutes[i]
            all_substitutes = [[int(c)] for c in lev_i]
        else:
            lev_i = []
            for all_sub in all_substitutes:
                for j in substitutes[i]:
                    lev_i.append(all_sub + [int(j)])
            all_substitutes = lev_i

    # all substitutes  list of list of token-id (all candidates)
    c_loss = nn.CrossEntropyLoss(reduction='none')
    word_list = []
    # all_substitutes = all_substitutes[:24]
    all_substitutes = torch.tensor(all_substitutes)  # [ N, L ]
    all_substitutes = all_substitutes[:24].to(device)
    # print(substitutes.size(), all_substitutes.size())
    N, L = all_substitutes.size()
    word_predictions = mlm_model(all_substitutes)[0]  # N L vocab-size
    ppl = c_loss(word_predictions.view(N * L, -1), all_substitutes.view(-1))  # [ N*L ]
    ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1))  # N
    _, word_list = torch.sort(ppl)
    word_list = [all_substitutes[i] for i in word_list]
    final_words = []
    for word in word_list:
        tokens = [tokenizer._convert_id_to_token(int(i)) for i in word]
        text = tokenizer.convert_tokens_to_string(tokens)
        final_words.append(text)
    return final_words
