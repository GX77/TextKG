"""
This script handles the training process.
"""

import argparse
import math
import time
import random
import numpy as np
import os
import subprocess
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import pdb
import traceback
from bdb import BdbQuit

from rtransformer.recursive_caption_dataset import single_sentence_collate, prepare_batch_inputs
from rtransformer.recursive_caption_dataset import RecursiveCaptionDataset as RCDataset
from rtransformer.model import TextKG
from rtransformer.optimization import BertAdam, EMA
from translator import Translator
from translate import run_translate
from utils_func import save_parsed_args_to_json, save_json, load_json, count_parameters, merge_dicts
from easydict import EasyDict as EDict
from tensorboardX import SummaryWriter
import logging
import string
from caption_eval import evaluate

logger = logging.getLogger(__name__)


def cal_performance(pred, gold):
    pred = pred.max(2)[1].contiguous().view(-1)
    gold = gold.contiguous().view(-1)
    valid_label_mask = gold.ne(RCDataset.IGNORE)
    pred_correct_mask = pred.eq(gold)
    n_correct = pred_correct_mask.masked_select(valid_label_mask).sum().item()
    return n_correct


def train_epoch(model, training_data, optimizer, ema, device, opt, writer, epoch):
    model.train()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    torch.autograd.set_detect_anomaly(True)
    for batch_idx, batch in tqdm(enumerate(training_data), mininterval=2,desc="  Training =>", total=len(training_data)):
        niter = epoch * len(training_data) + batch_idx
        writer.add_scalar("Train/LearningRate", float(optimizer.param_groups[0]["lr"]), niter)

        # prepare data
        batched_data = prepare_batch_inputs(batch[0], device=device, non_blocking=opt.pin_memory)
        input_ids = batched_data["input_ids"]
        input_ids2 = batched_data["input_ids2"]
        video_features = batched_data["video_feature"]
        region_features = batched_data['region_feature']
        input_masks = batched_data["input_mask"]
        input_masks2 = batched_data["input_mask2"]
        token_type_ids = batched_data["token_type_ids"]
        token_type_ids2 = batched_data["token_type_ids2"]
        input_labels = batched_data["input_labels"]
        tags = batched_data["tag"]
        kg_masks = batched_data['kg_mask']
        l_mask = batched_data['l_mask']
        gu = batched_data['gu']

        # forward & backward
        optimizer.zero_grad()
        loss, pred_scores = model(input_ids,input_ids2, video_features, region_features,input_masks, input_masks2, token_type_ids, token_type_ids2,l_mask,input_labels,tags,kg_masks)
        # make it consistent with other configs
        pred_scores_list = [pred_scores]
        input_labels_list = [input_labels]

        loss.backward(loss.clone().detach())
        if opt.grad_clip != -1:  # enable, -1 == disable
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()

        # update model parameters with ema
        if ema is not None:
            ema(model, niter)

        # keep logs
        n_correct = 0
        n_word = 0
        for pred, gold in zip(pred_scores_list, input_labels_list):
            n_correct += cal_performance(pred, gold)
            valid_label_mask = gold.ne(RCDataset.IGNORE)
            n_word += valid_label_mask.sum().item()

        n_word_total += n_word
        n_word_correct += n_correct
        total_loss += loss.sum().item()

    torch.autograd.set_detect_anomaly(False)

    loss_per_word = 1.0 * total_loss / n_word_total
    accuracy = 1.0 * n_word_correct / n_word_total
    return loss_per_word, accuracy


def eval_epoch(model, validation_data, device, opt):
    """The same setting as training, where ground-truth word x_{t-1}
    is used to predict next word x_{t}, not realistic for real inference"""
    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc="  Validation =>"):
            # prepare data
            batched_data = prepare_batch_inputs(batch[0], device=device, non_blocking=opt.pin_memory)
            input_ids = batched_data["input_ids"]
            input_ids2 = batched_data["input_ids2"]
            video_features = batched_data["video_feature"]
            region_features = batched_data['region_feature']
            input_masks = batched_data["input_mask"]
            input_masks2 = batched_data["input_mask2"]
            token_type_ids = batched_data["token_type_ids"]
            token_type_ids2 = batched_data["token_type_ids2"]
            input_labels = batched_data["input_labels"]
            tags = batched_data["tag"]
            kg_masks = batched_data['kg_mask']
            l_mask = batched_data['l_mask']

            loss, pred_scores = model(input_ids,input_ids2, video_features, region_features,input_masks, input_masks2, token_type_ids, token_type_ids2,l_mask,input_labels, tags, kg_masks)
            pred_scores_list = [pred_scores]
            input_labels_list = [input_labels]

            # keep logs
            n_correct = 0
            n_word = 0
            for pred, gold in zip(pred_scores_list, input_labels_list):
                n_correct += cal_performance(pred, gold)
                valid_label_mask = gold.ne(RCDataset.IGNORE)
                n_word += valid_label_mask.sum().item()

            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss.sum().item()

    loss_per_word = 1.0 * total_loss / n_word_total
    accuracy = 1.0 * n_word_correct / n_word_total
    return loss_per_word, accuracy


def eval_language_metrics(checkpoint, eval_data_loader, opt, model=None, eval_mode="val"):
    """eval_mode can only be set to `val` here, as setting to `test` is cheating
    0, run inference
    1, Get METEOR, BLEU1-4, CIDEr scores
    2, Get vocab size, sentence length
    """
    translator = Translator(opt, checkpoint, model=model)
    json_res = run_translate(eval_data_loader, translator, opt=opt)
    res_filepath = os.path.abspath(opt.save_model + "_tmp_greedy_pred_{}.json".format(eval_mode))
    save_json(json_res, res_filepath, save_pretty=True)

    if opt.dset_name == "anet":
        reference_files_map = {
            "val": [os.path.join(opt.data_dir, e) for e in
                    ["anet_entities_val_1_para.json", "anet_entities_val_2_para.json"]],
            "test": [os.path.join(opt.data_dir, e) for e in
                     ["anet_entities_test_1_para.json", "anet_entities_test_2_para.json"]]}
    else:  # yc2
        reference_files_map = {"val": [os.path.join(opt.data_dir, "yc2_val_anet_format_para.json")]}

    # COCO language evaluation
    eval_references = reference_files_map[eval_mode]
    lang_filepath = res_filepath.replace(".json", "_lang.json")
    eval_cmd = ["python3", "para-evaluate.py", "-s", res_filepath, "-o", lang_filepath,
                "-v", "-r"] + eval_references
    subprocess.call(eval_cmd, cwd=opt.eval_tool_dir)

# ['python para-evaluate.py -s '/mnt/Pycharm_Remote/recurrent_transformer/src/results_deadline/anet_single_res_2021_09_10_17_00_09/model_tmp_greedy_pred_val.json' -o /mnt/Pycharm_Remote/recurrent_transformer/src/results_deadline/anet_single_res_2021_09_10_17_25_43/model_tmp_greedy_pred_val_lang.json -v -r '/mnt/Pycharm_Remote/recurrent_transformer/densevid_eval/anet_data/anet_entities_val_1_para.json' '/mnt/Pycharm_Remote/recurrent_transformer/densevid_eval/anet_data/anet_entities_val_2_para.json']
    # basic stats
    stat_filepath = res_filepath.replace(".json", "_stat.json")
    eval_stat_cmd = ["python3", "get_caption_stat.py", "-s", res_filepath, "-r", eval_references[0],
                     "-o", stat_filepath, "-v"]
    subprocess.call(eval_stat_cmd, cwd=opt.eval_tool_dir)

    # repetition evaluation
    #rep_filepath = res_filepath.replace(".json", "_rep.json")
    #eval_rep_cmd = ["python3", "evaluateRepetition.py", "-s", res_filepath, "-r", eval_references[0],
    #                "-o", rep_filepath]
    #subprocess.call(eval_rep_cmd, cwd=opt.eval_tool_dir)

    # save results
    logger.info("Finished eval {}.".format(eval_mode))
    metric_filepaths = [lang_filepath, stat_filepath]
    all_metrics = merge_dicts([load_json(e) for e in metric_filepaths])

    all_metrics_filepath = res_filepath.replace(".json", "_all_metrics.json")
    save_json(all_metrics, all_metrics_filepath, save_pretty=True)
    return all_metrics, [res_filepath, all_metrics_filepath]


def train(model, training_data, validation_data, device, opt):
    model = model.to(device)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    if opt.ema_decay != -1:
        ema = EMA(opt.ema_decay)
        for name, p in model.named_parameters():
            if p.requires_grad:
                ema.register(name, p.data)
    else:
        ema = None

    num_train_optimization_steps = len(training_data) * opt.n_epoch
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=opt.lr,
                         warmup=opt.lr_warmup_proportion,
                         t_total=num_train_optimization_steps,
                         schedule="warmup_linear")

    writer = SummaryWriter(opt.res_dir)
    log_train_file = None
    log_valid_file = None

    if opt.log:
        log_train_file = opt.log + ".train.log"
        log_valid_file = opt.log + ".valid.log"

        logger.info("Training performance will be written to file: {} and {}".format(
            log_train_file, log_valid_file))

        with open(log_train_file, "w") as log_tf, open(log_valid_file, "w") as log_vf:
            log_tf.write("epoch,loss,ppl,accuracy\n")
            log_vf.write("epoch,loss,ppl,accuracy,METEOR,BLEU@4,CIDEr\n")

    prev_best_score = 0.
    es_cnt = 0
    for epoch_i in range(opt.n_epoch):
        logger.info("[Epoch {}]".format(epoch_i))
        # schedule sampling prob update, TODO not implemented yet

        start = time.time()
        if ema is not None and epoch_i != 0:  # use normal parameters for training, not EMA model
            ema.resume(model)
        train_loss, train_acc = train_epoch(
            model, training_data, optimizer, ema, device, opt, writer, epoch_i)
        logger.info("[Training]  ppl: {ppl: 8.5f}, accuracy: {acc:3.3f} %, elapse {elapse:3.3f} min"
                    .format(ppl=math.exp(min(train_loss, 100)), acc=100*train_acc, elapse=(time.time()-start)/60.))
        niter = (epoch_i + 1) * len(training_data)  # number of bart
        writer.add_scalar("Train/Acc", train_acc, niter)
        writer.add_scalar("Train/Loss", train_loss, niter)

        start = time.time()

        # Note here GT words are used to predicted next words, the same as training case!
        if ema is not None:
            ema.assign(model)  # EMA model
        val_loss, val_acc = eval_epoch(model, validation_data, device, opt)
        logger.info("[Val]  ppl: {ppl: 8.5f}, accuracy: {acc:3.3f} %, elapse {elapse:3.3f} min"
                    .format(ppl=math.exp(min(val_loss, 100)), acc=100*val_acc, elapse=(time.time()-start)/60.))
        writer.add_scalar("Val/Acc", val_acc, niter)
        writer.add_scalar("Val/Loss", val_loss, niter)

        # Note here we use greedy generated words to predicted next words, the true inference situation.
        checkpoint = {
            "model": model.state_dict(),  # EMA model
            "model_cfg": model.config,
            "opt": opt,
            "epoch": epoch_i}
        if epoch_i > 6 or epoch_i == 0:
            val_greedy_output, filepaths = eval_language_metrics(checkpoint, validation_data, opt, eval_mode="val", model=model)
            cider = val_greedy_output["CIDEr"]
            logger.info("[Val] METEOR {m:.2f} Bleu@4 {b:.2f} CIDEr {c:.2f}"
                        .format(m=val_greedy_output["METEOR"]*100,
                                b=val_greedy_output["Bleu_4"]*100,
                                c=val_greedy_output["CIDEr"]*100))
            writer.add_scalar("Val/METEOR", val_greedy_output["METEOR"]*100, niter)
            writer.add_scalar("Val/Bleu_4", val_greedy_output["Bleu_4"]*100, niter)
            writer.add_scalar("Val/CIDEr", val_greedy_output["CIDEr"]*100, niter)

            if opt.save_mode == "best":
                model_name = opt.save_model + ".pt"
                if cider > prev_best_score:
                    es_cnt = 0
                    prev_best_score = cider
                    torch.save(checkpoint, model_name)
                    new_filepaths = [e.replace("tmp", "best") for e in filepaths]
                    for src, tgt in zip(filepaths, new_filepaths):
                        os.renames(src, tgt)
                else:
                    es_cnt += 1
                    if es_cnt > opt.max_es_cnt:  # early stop
                        logger.info("Early stop at {} with CIDEr {}".format(epoch_i, prev_best_score))
                        break
            cfg_name = opt.save_model + ".cfg.json"
            save_parsed_args_to_json(opt, cfg_name)

            if log_train_file and log_valid_file:
                with open(log_train_file, "a") as log_tf, open(log_valid_file, "a") as log_vf:
                    log_tf.write("{epoch},{loss: 8.5f},{ppl: 8.5f},{acc:3.3f}\n".format(
                        epoch=epoch_i, loss=train_loss, ppl=math.exp(min(train_loss, 100)), acc=100*train_acc))
                    log_vf.write("{epoch},{loss: 8.5f},{ppl: 8.5f},{acc:3.3f},{m:.2f},{b:.2f},{c:.2f}\n".format(
                        epoch=epoch_i, loss=val_loss, ppl=math.exp(min(val_loss, 100)), acc=100*val_acc,
                        m=val_greedy_output["METEOR"]*100,
                        b=val_greedy_output["Bleu_4"]*100,
                        c=val_greedy_output["CIDEr"]*100))
    writer.close()


def get_args():
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    """parse and preprocess cmd line args"""
    parser = argparse.ArgumentParser()
    parser.add_argument("---num_workers", type=int, default=4, help="num subprocesses used to load the data, 0: use main process")
    parser.add_argument("--batch_size", type=int, default=5, help="training batch size")
    parser.add_argument("--res_root_dir", type=str, default="../0626/result_github/", help="dir to containing all the results")
    parser.add_argument("--word2idx_path", type=str, default="./cache/yc2_word2idx_3_add7.json")  # new_
    parser.add_argument("--glove_path", type=str, default='./cache/yc2_vocab_glove_3_add7.pt', help="extracted GloVe vectors")
    parser.add_argument("--data_dir", default='../densevid_eval/yc2_data', help="dir containing the splits data files")
    parser.add_argument("--recurrent", type=str, default=False)
    parser.add_argument("--name", type=str, default='yc2_single', help="the gailv of self word")
    parser.add_argument("--seed", default=2027, type=int)

    parser.add_argument("--type_vocab_size", type=int, default=5, help="video as 0, text as 1")
    parser.add_argument("--freeze_glove", default=False, help="do not train GloVe vectors")
    parser.add_argument("--dset_name", type=str, default="yc2", choices=["anet", "yc2"], help="Name of the dataset, will affect data loader, evaluation, etc")
    parser.add_argument("--video_feature_dir", default='/mnt/bn/gxnas-kg-lq/Dataset/YouCook2/rt_yc2_feat/trainval/', help="dir containing the video features")
    parser.add_argument("--region_feature_dir", default='/mnt/bn/gxnas-kg-lq/Dataset/YouCook2/rt_yc2_region/trainval/')
    parser.add_argument("--region_feature_300dir", type=str, default='/mnt/bn/gxnas-kg-lq/Dataset/YouCook2/youcook2/region_yc2feature_300d/', help="the gailv of self word")
    parser.add_argument("--region_kg_dir", default='/mnt/bn/gxnas-kg-lq/Dataset/YouCook2/youcook2/name_tag_index.json')
    parser.add_argument("--asr_dir", default='/mnt/bn/gxnas-kg-lq/Dataset/YouCook2/youcook2/yc2_asr.json')
    parser.add_argument("--asrkg_dir", default='/mnt/bn/gxnas-kg-lq/Dataset/YouCook2/youcook2/asr_kg_index_old_add_rel.json')
    parser.add_argument("--outkg_dir", default='/mnt/bn/gxnas-kg-lq/Dataset/YouCook2/youcook2/xizhi_dic_index.json')
    parser.add_argument("--v_duration_file", default='/mnt/bn/gxnas-kg-lq/Dataset/YouCook2/rt_yc2_feat/yc2_duration_frame.csv',help="filepath to the duration file")

    parser.add_argument("--n_epoch", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--untied", action="store_true", help="Run untied model")
    parser.add_argument("--xl", action="store_true", help="transformer xl model, when specified, will automatically set recurrent = True,since the data loading part is the same")
    parser.add_argument("--xl_grad", action="store_true", help="enable back-propagation for xl model, only useful when `-xl` flag is enabled.Note, the original transformerXL model does not allow back-propagation.")
    parser.add_argument("--mtrans", action="store_true", help="Masked transformer model for single sentence generation")
    parser.add_argument("--save_model", default="model")

    # model config
    parser.add_argument("--eval_tool_dir", type=str, default="../densevid_eval")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--intermediate_size", type=int, default=768)
    parser.add_argument("--vocab_size", type=int, help="number of words in the vocabulary")
    parser.add_argument("--word_vec_size", type=int, default=300)
    parser.add_argument("--video_feature_size", type=int, default=3072, help="2048 appearance + 1024 flow")
    parser.add_argument("--max_v_len", type=int, default=100, help="max length of video feature")
    parser.add_argument("--max_t_len", type=int, default=22,  help="max length of text (sentence or paragraph), 30 for anet, 20 for yc2")
    parser.add_argument("--max_n_sen", type=int, default=12,   help="for recurrent, max number of sentences, 6 for anet, 10 for yc2")
    parser.add_argument("--max_a_len", type=int, default=300, help="asr length")
    parser.add_argument("--layer_norm_eps", type=float, default=1e-12)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of transformer layers")
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.1)
    parser.add_argument("--num_attention_heads", type=int, default=12)
    parser.add_argument("--memory_dropout_prob", type=float, default=0.1)
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--share_wd_cls_weight", action="store_true",help="share weight matrix of the word embedding with the final classifier, ")
    # training config -- learning rate
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10% of training.")
    parser.add_argument("--grad_clip", type=float, default=1, help="clip gradient, -1 == disable")
    parser.add_argument("--ema_decay", default=0.9999, type=float,
                        help="Use exponential moving average at training, float in (0, 1) and -1: do not use.  "
                             "ema_param = new_param * ema_decay + (1-ema_decay) * last_param")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Use soft target instead of one-hot hard target")
    parser.add_argument("--max_es_cnt", type=int, default=10, help="stop if the model is not improving for max_es_cnt max_es_cnt")
    parser.add_argument("--val_batch_size", type=int, default=4, help="inference batch size")
    parser.add_argument("--use_beam", action="store_true", help="use beam search, otherwise greedy search")
    parser.add_argument("--beam_size", type=int, default=2, help="beam size")
    parser.add_argument("--n_best", type=int, default=1, help="stop searching when get n_best from beam search")
    # others
    parser.add_argument("--no_pin_memory", default=True,
                        help="Don't use pin_memory=True for dataloader. "
                             "ref: https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/4")
    parser.add_argument("--exp_id", type=str, default="init", help="id of the current run")
    parser.add_argument("--save_mode", type=str, choices=["all", "best"], default="best", help="all: save models at each epoch; best: only save the best model")
    parser.add_argument("--no_cuda", action="store_true", help="run on cpu")
    parser.add_argument("--debug", action="store_true")


    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    opt.recurrent = True if opt.xl else opt.recurrent
    assert not (opt.recurrent and opt.untied), "cannot be True for both"
    assert not (opt.recurrent and opt.mtrans), "cannot be True for both"
    assert not (opt.untied and opt.mtrans), "cannot be True for both"
    if opt.xl_grad:
        assert opt.xl, "`-xl` flag must be set when using `-xl_grad`."

    ran_str = str(opt.seed) +''.join(random.sample(string.ascii_letters + string.digits, 3))
    # make paths
    opt.res_dir = os.path.join(
        opt.res_root_dir, "_".join([opt.name,ran_str]))

    if os.path.exists(opt.res_dir) and os.listdir(opt.res_dir):
        raise ValueError("File exists {}".format(opt.res_dir))
    elif not os.path.exists(opt.res_dir):
        os.makedirs(opt.res_dir)

    opt.log = os.path.join(opt.res_dir, opt.save_model)
    opt.save_model = os.path.join(opt.res_dir, opt.save_model)
    opt.pin_memory = not opt.no_pin_memory

    if opt.share_wd_cls_weight:
        assert opt.word_vec_size == opt.hidden_size, \
            "hidden size has to be the same as word embedding size when " \
            "sharing the word embedding weight and the final classifier weight"
    return opt


def main():
    opt = get_args()

    # random seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True

    train_dataset = RCDataset(
        dset_name=opt.dset_name,
        data_dir=opt.data_dir, video_feature_dir=opt.video_feature_dir,region_feature_dir=opt.region_feature_dir,
        region_kg_dir = opt.region_kg_dir, asr_path = opt.asr_dir,asrkg_path=opt.asrkg_dir, outkg_path = opt.outkg_dir,
        duration_file=opt.v_duration_file,
        word2idx_path=opt.word2idx_path, max_t_len=opt.max_t_len,
        max_v_len=opt.max_v_len, max_n_sen=opt.max_n_sen, max_a_len=opt.max_a_len, region_feature_300dir=opt.region_feature_300dir, mode="train",
        recurrent=opt.recurrent, untied=opt.untied or opt.mtrans)
    # add 10 at max_n_sen to make the inference stage use all the segments
    val_dataset = RCDataset(
        dset_name=opt.dset_name,
        data_dir=opt.data_dir, video_feature_dir=opt.video_feature_dir,region_feature_dir=opt.region_feature_dir,
        region_kg_dir = opt.region_kg_dir, asr_path = opt.asr_dir,asrkg_path = opt.asrkg_dir,outkg_path = opt.outkg_dir,
        duration_file=opt.v_duration_file,
        word2idx_path=opt.word2idx_path, max_t_len=opt.max_t_len,
        max_v_len=opt.max_v_len, max_n_sen=opt.max_n_sen+10, max_a_len=opt.max_a_len, region_feature_300dir=opt.region_feature_300dir, mode="val",
        recurrent=opt.recurrent, untied=opt.untied or opt.mtrans)

    generator = torch.Generator()
    generator.manual_seed(opt.seed)
    train_loader = DataLoader(train_dataset, collate_fn=single_sentence_collate,
                              batch_size=opt.batch_size, shuffle=True,
                              num_workers=opt.num_workers, pin_memory=opt.pin_memory,generator=generator)
    val_loader = DataLoader(val_dataset, collate_fn=single_sentence_collate,
                            batch_size=opt.val_batch_size, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=opt.pin_memory)

    opt.vocab_size = len(train_dataset.word2idx)

    device = torch.device("cuda" if opt.cuda else "cpu")
    rt_config = EDict(
        xl_grad=opt.xl_grad,  # enable back-propagation for transformerXL model
        hidden_size=opt.hidden_size,
        intermediate_size=opt.intermediate_size,  # after each self attention
        vocab_size=opt.vocab_size,  # get from word2idx
        word_vec_size=opt.word_vec_size,
        video_feature_size=opt.video_feature_size,
        max_position_embeddings=opt.max_v_len + opt.max_t_len,  # get from max_seq_len
        max_v_len=opt.max_v_len,  # max length of the videos
        max_t_len=opt.max_t_len,  # max length of the text
        type_vocab_size=opt.type_vocab_size,
        layer_norm_eps=opt.layer_norm_eps,  # bert layernorm
        hidden_dropout_prob=opt.hidden_dropout_prob,  # applies everywhere except attention
        num_hidden_layers=opt.num_hidden_layers,  # number of transformer layers
        num_attention_heads=opt.num_attention_heads,
        attention_probs_dropout_prob=opt.attention_probs_dropout_prob,  # applies only to self attention
        memory_dropout_prob=opt.memory_dropout_prob,
        initializer_range=opt.initializer_range,
        label_smoothing=opt.label_smoothing,
        share_wd_cls_weight=opt.share_wd_cls_weight
    )

    model = TextKG(rt_config)

    if opt.glove_path is not None:
        if hasattr(model, "embeddings"):
            logger.info("Load GloVe as word embedding")
            model.embeddings.set_pretrained_embedding(torch.from_numpy(torch.load(opt.glove_path)).float(), freeze=opt.freeze_glove)
        else:
            logger.warning("This model has no embeddings, cannot load glove vectors into the model")

    count_parameters(model)
    if hasattr(model, "embeddings") and hasattr(model.embeddings, "word_embeddings"):
        count_parameters(model.embeddings.word_embeddings)

    train(model, train_loader, val_loader, device, opt)



if __name__ == "__main__":
    try:
        main()
    except BdbQuit:
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        print('')
        pdb.post_mortem()
        sys.exit(1)

