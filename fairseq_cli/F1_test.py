#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import argparse
import logging
import math
import os
import sys
from typing import Dict, Optional, Any, List, Tuple, Callable
from tensorboardX import SummaryWriter

import numpy as np
import torch
import time

from fairseq import (
    checkpoint_utils,
    distributed_utils,
    options,
    quantization_utils,
    tasks,
    utils,
)
from fairseq.data import iterators
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import meters, metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from omegaconf import DictConfig
from fairseq.trainer import Trainer
from helpers.nltk_tree import build_nltktree_only_bracket
from nltk import Tree

import re
from tqdm import tqdm
import matplotlib.pylab as plt

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")

def weights_init(mod):
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        mod.weight.data.normal_(0.0,0.02)
    elif classname.find('BatchNorm')!= -1:
        mod.weight.data.normal_(1.0,0.02)
        mod.bias.data.fill_(0)


def create_sentence(src, src_dict, remove_bpe=True):
    res = ""
    if isinstance(src, torch.Tensor):
        if src.is_cuda:
            src = src.detach().cpu()
    src_len = src.shape[0]
    for i in range(src_len):
        tok = src_dict[src[i]]
        if src[i] == 18:
            tok = "''"
        # if tok == '.':
        #     tok = ','
        if tok == '–': # <- STARNGE BUG -->
            tok = '--'
        if tok == '>>':
            tok = "--"
        # if ascii(tok)=="'\\u2013'": # for tok = "-"
        #     tok = "--"

        if not remove_bpe:
            if i == src_len - 1:
                res += tok
            else:
                res += tok + " "
        else:
            if tok.endswith("@@"):
                res += tok[:-2]
            else:
                if i == src_len - 1:
                    res += tok
                else:
                    res += tok + " "

    # remove start / end symbol > still have other symbols ?
    remove_start = 0
    remove_end = 0
    # if res.startswith("<s>"):
    #     res = res[3:]
    #     remove_start = 1
    # if res.endswith("</s>"):
    #     res = res[:-5]
    #     remove_end = 1
    return res, remove_start, remove_end


def remove_bpe(sentense, distance):
    if not isinstance(sentense, list):
        print("except type of arg sentense to be len !")
        return
    sent_len = len(sentense)
    dist_wo_bpe = []
    sent_wo_bpe = ""
    for i in range(sent_len):
        token = sentense[i]
        if not token.endswith("@@"):
            sent_wo_bpe += token
            if i != sent_len - 1:
                sent_wo_bpe += " "
            if i < sent_len - 1:
                dist = distance[i]
                dist_wo_bpe.append(dist)
        else:
            sent_wo_bpe += token[:-2]
    return sent_wo_bpe, dist_wo_bpe


def process_str_tree(str_tree):
    pat = re.compile('<[^>]+>')
    res = pat.sub("", str_tree)
    return re.sub('[ |\n]+', ' ', res)


def tree2sameNodes(tree, node_name):
    """
    modify a tree to share same node name, leaves remain unchanged
    :param tree: nltk.Tree
    :param node_name: modified nodes name
    :return: a modified tree
    """
    if isinstance(tree, Tree):
        tree.set_label(node_name)
        for child in tree:
            tree2sameNodes(child, node_name)
    else:
        return


def evalb_test(evalb_path, pred_pth, gt_pth, out_pth, e=10000):
    """
    :param evalb_path : exe path
    :param pred_pth:  prediction txt file
    :param gt_pth: ground_truth txt file
    :param out_pth: output file of EVLB
    :param e: max tolerated error
    :return: F1-score
    """
    param_pth = os.path.join(evalb_path, "sample/sample.prm")
    evalb_exe_pth = os.path.join(evalb_path, "evalb")
    cmd = "{} -p {} -e {} {} {} > {}".format(
        evalb_exe_pth,
        param_pth,
        e,
        pred_pth,
        gt_pth,
        out_pth
    )

    os.system(cmd)


    f = open(out_pth)
    for line in f:
        match = re.match(r"Bracketing Recall\s+=\s+(\d+\.\d+)", line)
        if match:
            recall = float(match.group(1))
        match = re.match(r"Bracketing Precision\s+=\s+(\d+\.\d+)", line)
        if match:
            precision = float(match.group(1))
        match = re.match(r"Bracketing FMeasure\s+=\s+(\d+\.\d+)", line)
        if match:
            fscore = float(match.group(1))
            break

    res = {
        "recall" : recall,
        "precision" : precision,
        "fscore" : fscore
    }

    return res


def main(cfg: DictConfig) -> None:

    # ---------------- some hyper params ----------------#
    device = 2
    # epc_list = range(5, 70, 5)
    subset_name = 'valid'

    root = '/home/jskai/workspace/fairseq'
    checkpoint_pth = 'experiments/iwslt14de-en/log/structformer-lm_scaler0.47-seed2-conv_attn_bpe256/checkpoint_avg5.pt'
    save_pth = 'experiments/iwslt14de-en/log/structformer-lm_scaler0.47-seed2-conv_attn_bpe256/bpe/'
    checkpoint_pth = os.path.join(root, checkpoint_pth)
    save_pth_r = os.path.join(root, save_pth)
    # path = "/home/htxue/data/Distance-Transformer/distance-transformer/src_data/stanford-corenlp-full-2018-10-05"
    gt_done = 0
    # invalid_sent_id = [2023]
    k = 1  # part of the dataset, size = dataset_size // k
    evalb_pth = "EVALB/"
    src_path = root + '/distance_prior/iwslt14de2en/' + subset_name + '/'
    bpe_path = '/home/jskai/workspace/fairseq/bpe-prior/iwslt14de2en/' + subset_name + '/'
    # ---------------------------------------------------#

    if os.path.exists(os.path.join(save_pth_r, "tree_gt.txt")):
        gt_done = 1

    if isinstance(cfg, argparse.Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)

    assert (
        cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"
    metrics.reset()

    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    if distributed_utils.is_master(cfg.distributed_training):
        checkpoint_utils.verify_checkpoint_directory(cfg.checkpoint.save_dir)

    # Print args
    logger.info(cfg)

    # Setup task, e.g., translation, language modeling, etc.

    task = tasks.setup_task(cfg.task)
    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in ['train', 'valid']:
        task.load_dataset(valid_sub_split, combine=False, epoch=1)

    assert cfg.criterion, "Please specify criterion to train a model"

    # Build model and criterion
    model = task.build_model(cfg.model)
    # model.apply(weights_init)

    dataset = task.dataset(subset_name)
    world_dict = dataset.src_dict

    save_pth = save_pth_r + "/" + "{}_interval{}".format(subset_name, k) + "/"

    if not os.path.exists(save_pth):
        os.makedirs(save_pth)
    if not os.path.exists(save_pth):
        os.mkdir(save_pth)

    # import pdb
    # pdb.set_trace()
    gt_done = False
    if os.path.exists(os.path.join(save_pth, "tree_gt.txt")):
        gt_done = True
    if not gt_done:
        f_gt = open(os.path.join(save_pth, 'tree_gt.txt'), "w")
        dataset_size = dataset.__len__()
        print(dataset_size)
        for sent_id in range(dataset_size//k):
            # print(sent_id)
            # if sent_id >= 20:
            #     break
            # if sent_id in invalid_sent_id:
            #     continue

            item = dataset[sent_id]
            src_tokens = item['source']
            src_tokens = src_tokens.unsqueeze(0)

            dist_caled = np.load(src_path + '{}.npy'.format(sent_id))
            sent, rs, re = create_sentence(src=src_tokens[0], src_dict=world_dict, remove_bpe=False)
            sent_list = sent.split(" ")
            tree_gld = build_nltktree_only_bracket(dist_caled, sent_list)
            # tree_gld.pretty_print()

            tree2sameNodes(tree_gld, "Node")

            tree_str = process_str_tree(str(tree_gld))

            f_gt.write(tree_str + "\n")

        f_gt.close()

    F_score_y = []
    F_score_x = []

# for epc in epc_list:
    dataset_size = dataset.__len__()
    print("[loading] ---checkpoint:{}---".format(checkpoint_pth))
    tmp = torch.load(checkpoint_pth)['model']
    model.load_state_dict(tmp)
    model.eval()

    # import pdb
    # pdb.set_trace()
    pred_done = False
    if os.path.exists(os.path.join(save_pth, "pred.txt")):
        pred_done = True
    if not pred_done:
        f_pred = open(os.path.join(save_pth, "pred.txt"), "w")

        for sent_id in range(dataset_size//k):
            # print(sent_id)
            # if sent_id >= 20:
            #     break
            # if sent_id in invalid_sent_id:
            #     continue
            time_st = time.time()

            item = dataset[sent_id]
            src_tokens = item['source']
            src_tokens = src_tokens.unsqueeze(0)

            sent_bpe, rs, re = create_sentence(src=src_tokens[0], src_dict=world_dict, remove_bpe=False)
            sent_list = sent_bpe.split(" ")

            # get distances
            # import pdb
            # pdb.set_trace()
            pos = torch.arange(src_tokens.size(1))[None, :]
            
            bpe_caled = np.load(bpe_path + '{}.npy'.format(sent_id))
            bpe_embedding = model.encoder.embed_bpe(torch.tensor(bpe_caled))
            h, _ = model.encoder.forward_embedding(src_tokens)
            dist, height = model.encoder.parse(src_tokens, pos, h)
            dist = dist[:,:-1]
            dist = dist.squeeze(0)
            dist_list = [i.item() for i in dist]

            tree_pred = build_nltktree_only_bracket(dist_list, sent_list)
            # tree_pred.pretty_print()

            tree2sameNodes(tree_pred, "Node")

            tree_pred_str = process_str_tree(str(tree_pred))

            f_pred.write("(Node " + tree_pred_str + ")")
            f_pred.write("\n")

        f_pred.close()

    res = evalb_test(os.path.join(root, evalb_pth),
                        os.path.join(save_pth, "pred.txt"),
                        os.path.join(save_pth, "tree_gt.txt"),
                        os.path.join(save_pth, "tmp.txt"),
                        e=10000)
    print("The test score is as follows:")
    print(res)
    # F_score_y.append(res['fscore'])
    # F_score_x.append(epc)
    # plt.plot(F_score_x, F_score_y, "ob:")
    # plt.savefig(save_pth + "curve_{}.png".format(mode))
    # plt.close()


def cal_F1_main(
    modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
) -> None:
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)

    cfg = convert_namespace_to_omegaconf(args)

    print(cfg)
    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(cfg, main)
    else:
        distributed_utils.call_main(cfg, main)


if __name__ == "__main__":
    cal_F1_main()
