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

import numpy as np
import torch
import time
import threading
from multiprocessing import Pool
import multiprocessing
import concurrent.futures

from fairseq import (
    checkpoint_utils,
    distributed_utils,
    options,
    tasks,
    utils,
)


from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import meters, metrics, progress_bar
from omegaconf import DictConfig
from tqdm import tqdm


import re
import nltk
import matplotlib.pylab as plt

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")


def thread_function(sent_id):
    # import pdb
    # pdb.set_trace()
    item = dataset[sent_id]
    src_tokens = item['source']

    if isinstance(src_tokens, torch.Tensor):
        if src_tokens.is_cuda:
            src_tokens = src_tokens.detach().cpu()

    num_token = src_tokens.shape[0]
    raw_word_list = [word_dict[src_tokens[i]] for i in range(num_token)]

    bpe_list = []
    i = 0

    while i < len(raw_word_list):
        j = i
        while raw_word_list[i].endswith("@@"):
            bpe_list.append(2)
            i = i + 1
        if j < i:
            bpe_list.append(2)
        else:
            bpe_list.append(0)
        i = i + 1

    if sent_id % 1000 == 0:
        print("-" * 100)
        print("sent_id:", sent_id)
        print("raw_sentence:\n", raw_word_list)
        print("bpe:\n", bpe_list)
        print("-" * 100)

    np.save(split_saving + "/{}.npy".format(sent_id), bpe_list)



def main(cfg: DictConfig) -> None:
    global nlp
    global dataset
    global word_dict
    global split_saving

    # ---------------- some hyper params ----------------#
    root = "/home/jskai/workspace/fairseq"
    save_path = os.path.join(root, "bpe-prior", "wmt14en2de1")
    src_tgt = 'src'
    # ----------------  params domains ----------------#
    assert src_tgt in ['src', 'tgt']

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

    # Setup task, e.g., translation, language modeling, etc.

    task = tasks.setup_task(cfg.task)


    for split in ['train', 'test', 'valid']:
        # ---------------------------------------------------#

        # ----------------- make necessary dir ---------- #
        split_saving = os.path.join(save_path, split)
        if not os.path.exists(split_saving):
            os.makedirs(split_saving)

        # Load valid dataset (we load training data below, based on the latest checkpoint)
        task.load_dataset(split, combine=False, epoch=1)

        dataset = task.dataset(split)
        word_dict = dataset.src_dict if src_tgt == 'src' else dataset.tgt_dict

        # thread_function(0)
        poolsize = 1000
        pool = multiprocessing.Pool(processes=poolsize)
        for i, sent_id in enumerate(tqdm(range(dataset.__len__()))):
            pool.apply_async(thread_function, (sent_id,))
            # if not os.path.exists(os.path.join(split_saving,str(i)+".npy")):
            #     print(sent_id)
            # # if i>=200000 and i<240000:
            #     thread_function(sent_id)
        pool.close()
        pool.join()

        dirs1 = os.listdir(split_saving)
        cnts1 = set()
        for dir in dirs1:
            cnts1.add(int(dir[:-4]))
        print(split + ' files generated:' + str(len(cnts1)) + ', total file:' + str(len(dataset)))
        assert len(dataset) == len(cnts1)
        print("ending for execution")


def cal_bpe_prior_main(
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
    cal_bpe_prior_main()
