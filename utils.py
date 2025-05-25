#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def pad_sents(sents, pad_token):
    sents_padded = []

    max_len = max([len(sent) for sent in sents])
    for sent in sents:
      sents_padded.append(sent + [pad_token]*(max_len-len(sent)))

    return sents_padded


def read_corpus(file_path):
    src = []
    tgt = []

    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            line = line[:-1]
            s = line.split(" ")
            src.append(s[:-1])
            tgt.append(s[1:])

    return src, tgt


def batch_iter(data, batch_size, shuffle=False):
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents


