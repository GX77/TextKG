# -*- coding: utf-8 -*-
# @Time    : 2022/11/21 18:44
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : caption_test.py
import os
import logging
from collections import defaultdict
from tqdm import tqdm

import torch.distributed as dist
import torch.utils.data

from utils.train_utils import gather_object_multiple_gpu, get_timestamp, CudaPreFetcher
from utils.json import save_json, load_json
from utils.train_utils import Timer

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

logger = logging.getLogger(__name__)


class Translator(object):
    """Load with trained model and handle the beam search"""

    def __init__(self, checkpoint, model=None):
        self.max_v_len = checkpoint['opt'].max_v_len
        self.max_t_len = checkpoint['opt'].max_t_len
        self.PAD = 0
        self.BOS = 4

        self.model = model
        self.model.eval()

        self.timer = Timer(synchronize=True, history_size=500, precision=3)

    def translate_batch_single_sentence_greedy(self, inputs, model):
        inputs_ids = inputs["input_ids"]
        input_masks = inputs["input_mask"]
        # max_t_len = self.max_t_len
        max_t_len = 21
        inputs_ids[:, :] = 0.
        input_masks[:, :] = 0.
        assert torch.sum(input_masks[:, :]) == 0, "Initially, all text tokens should be masked"
        bsz = len(inputs_ids)
        next_symbols = torch.IntTensor([self.BOS] * bsz)  # (N, )

        self.timer.reset()
        for dec_idx in range(max_t_len):
            inputs_ids[:, dec_idx] = next_symbols.clone()
            input_masks[:, dec_idx] = 1
            outputs = model(inputs)
            pred_scores = outputs["prediction_scores"]
            # pred_scores[:, :, 49406] = -1e10
            next_words = pred_scores[:, dec_idx].max(1)[1]  # TODO / NOTE changed
            next_symbols = next_words.cpu()
            if "visual_output" in outputs:
                inputs["visual_output"] = outputs["visual_output"]
            else:
                logger.debug("visual_output is not in the output of model, this may slow down the caption test")
        self.timer(stage_name="inference")
        logger.debug(f"inference toke {self.timer.get_info()['average']['inference']} ms")
        return inputs_ids

    def translate_batch(self, model_inputs):
        """while we used *_list as the input names, they could be non-list for single sentence decoding case"""
        return self.translate_batch_single_sentence_greedy(model_inputs, self.model)


def convert_ids_to_sentence(tokens):
    from .clip.clip import _tokenizer
    text = _tokenizer.decode(tokens)
    text_list = text.split(" ")
    new = []
    for i in range(len(text_list)):
        if i == 0:
            new.append(text_list[i].split(">")[-1])
        elif "<|endoftext|>" in text_list[i]:
            break
        else:
            new.append(text_list[i])
    return " ".join(new)


def run_translate(data_loader, translator, epoch, opt):
    # submission template
    batch_res = {"version": "VERSION 1.0",
                 "results": defaultdict(list),
                 "external_data": {"used": "true", "details": "ay"}}
    for bid, batch in enumerate(tqdm(data_loader,
                                     dynamic_ncols=True,
                                     disable=dist.is_initialized() and dist.get_rank() != 0)):
        if torch.cuda.is_available():
            batch = CudaPreFetcher.cuda(batch)
        dec_seq = translator.translate_batch(batch)

        # example_idx indicates which example is in the batch
        for example_idx, (cur_gen_sen, cur_meta) in enumerate(zip(dec_seq, batch['metadata'][1])):
            cur_data = {
                "sentence": convert_ids_to_sentence(cur_gen_sen.tolist()),
                "gt_sentence": cur_meta
            }
            # print(cur_data)
            batch_res["results"][batch['metadata'][0][example_idx].split("video")[-1]].append(cur_data)
    translator.timer.print()
    return batch_res


class EvalCap:
    def __init__(self, annos, rests, cls_tokenizer=PTBTokenizer,
                 use_scorers=['Bleu', 'METEOR', 'ROUGE_L', 'CIDEr']):  # ,'SPICE']):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.annos = annos
        self.rests = rests
        self.Tokenizer = cls_tokenizer
        self.use_scorers = use_scorers

    def evaluate(self):
        res = {}
        for r in self.rests:
            res[str(r['image_id'])] = [{'caption': r['caption']}]

        gts = {}
        for imgId in self.annos:
            gts[str(imgId)] = [{'caption': self.annos[imgId]}]

        # =================================================
        # Set up scorers
        # =================================================
        # print('tokenization...')
        tokenizer = self.Tokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)
        # gts = {k: v for k, v in gts.items() if k in res.keys()}
        # =================================================
        # Set up scorers
        # =================================================
        # print('setting up scorers...')
        use_scorers = self.use_scorers
        scorers = []
        if 'Bleu' in use_scorers:
            scorers.append((Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]))
        if 'METEOR' in use_scorers:
            scorers.append((Meteor(), "METEOR"))
        if 'ROUGE_L' in use_scorers:
            scorers.append((Rouge(), "ROUGE_L"))
        if 'CIDEr' in use_scorers:
            scorers.append((Cider(), "CIDEr"))
        if 'SPICE' in use_scorers:
            scorers.append((Spice(), "SPICE"))

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    # print("%s: %0.1f" % (m, sc*100))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                # print("%s: %0.1f" % (method, score*100))
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]


def evaluate(submission, reference):
    tokenizer = PTBTokenizer  # for English
    annos = reference
    data = submission['results']
    rests = []
    for name, value in data.items():
        rests.append({'image_id': str(name), 'caption': value[0]['sentence']})
    eval_cap = EvalCap(annos, rests, tokenizer)

    eval_cap.evaluate()

    all_score = {}
    for metric, score in eval_cap.eval.items():
        all_score[metric] = score
    return all_score

if __name__ == "__main__":
    ours = load_json("")
    gt = load_json("")
    evaluate(ours, gt)