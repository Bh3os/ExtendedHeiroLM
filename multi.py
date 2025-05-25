#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import sys
import time
from pathlib import Path

from models import load_model, Hypothesis
import numpy as np
from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm
from utils import read_corpus, batch_iter
from vocab import Vocab, VocabEntry

import torch
import torch.nn.utils

from parse import args
from sklearn.metrics import f1_score

from torch.utils.tensorboard import SummaryWriter

def read_sents(file_path):
    sents = []

    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            line = line[:-1]
            s = line.split(" ")
            sents.append(s)

    return sents

def evaluate_multishot(model, test_sents, device):
    bases = [0 for _ in range(4)]
    counts = [0 for _ in range(4)]
    for i in tqdm(range(len(test_sents))):
        sent = test_sents[i]
        for idx in range(len(sent)-1):
            max_shot = min(4, len(sent)-idx-1)
            for s in range(max_shot):
                bases[s] += 1
            for s in range(max_shot):
                tgt_idx = idx+s+1
                src = [sent[:tgt_idx]]
                prediction = model.predict_realtime(src, device)
                if prediction == sent[tgt_idx]:
                    counts[s] += 1
                else:
                    break
    return [counts[i]/bases[i] for i in range(4)]

def decode(args: Dict[str, str]):
    test_file = "data/"+args.dataset+"/"+args.test_file
    print("load test source sentences from [{}]".format(test_file), file=sys.stderr)
    test_sents = read_sents(test_file)

    model_path = "saved_models/"+str(args.embed_size)+"_"+str(args.hidden_size)+"_"+str(args.dropout)+"_"+args.dataset+"_"+args.model_type+"_"+args.model_path

    print("load model from {}".format(model_path), file=sys.stderr)
    model = load_model(model_path, args.model_type)

    use_cuda = args.cuda
    if isinstance(use_cuda, str):
        use_cuda = use_cuda.lower() == 'true'
    device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")
    model = model.to(device)

    #########
    #test_ppl = evaluate_ppl(model, test_data, batch_size=128, device=device)   # dev batch size can be a bit larger
    #test_accuracy, test_f1 = evaluate_accuracy_and_f1(model, test_data, batch_size=128, device=device)
    accs = evaluate_multishot(model,test_sents,device)
    for i in range(4):
        print(str(i+1)+" shot accuracy:",accs[i])

    #print('test: ppl %f, accuracy %f, f1 %f' % (test_ppl, test_accuracy, test_f1), file=sys.stderr)
    #with open(args.dataset+"_result.txt", "a+") as f:
    #    f.write(str(args.embed_size)+","+str(args.hidden_size)+","+str(args.dropout)+","+str(test_ppl)+","+str(test_accuracy)+","+str(test_f1)+"\n")



def main():
    decode(args)


if __name__ == '__main__':
    main()
