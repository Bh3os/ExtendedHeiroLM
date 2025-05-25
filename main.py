#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import sys
import time
from pathlib import Path

from models import get_model, load_model, Hypothesis
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

def evaluate_ppl(model, dev_data, batch_size, device):
    was_training = model.training
    model.eval()

    cum_loss = 0.
    cum_tgt_words = 0.

    with torch.no_grad():
        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            loss = -model(src_sents, tgt_sents, device).sum()

            cum_loss += loss.item()
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

    if was_training:
        model.train()

    return ppl

def evaluate_accuracy_and_f1(model, dev_data, batch_size, device):
    was_training = model.training
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            predictions, target_masks, target_padded, source_lengths = model.predict(src_sents, tgt_sents, device)
            
            # Flatten predictions and targets for accuracy calculation
            for i in range(len(src_sents)):
                seq_len = source_lengths[i]
                pred_seq = predictions[:seq_len, i].cpu().numpy()
                target_seq = target_padded[:seq_len, i].cpu().numpy()
                mask_seq = target_masks[:seq_len, i].cpu().numpy()
                
                # Only consider non-padded positions
                valid_positions = mask_seq == 1
                if valid_positions.sum() > 0:
                    all_predictions.extend(pred_seq[valid_positions])
                    all_targets.extend(target_seq[valid_positions])

    if was_training:
        model.train()

    accuracy = sum(p == t for p, t in zip(all_predictions, all_targets)) / len(all_predictions) if all_predictions else 0
    f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0) if all_targets else 0

    return accuracy, f1


def train(args):

    #### LOAD DATA

    train_file = "data/"+args.dataset+"/"+args.train_file
    train_data_src, train_data_tgt = read_corpus(train_file)

    print("loaded training set from", train_file)

    dev_file = "data/"+args.dataset+"/"+args.dev_file
    dev_data_src, dev_data_tgt = read_corpus(dev_file)

    print("loaded dev set from", dev_file)

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    train_batch_size = args.train_batch_size
    clip_grad = args.clip_grad
    valid_niter = args.valid_niter
    log_every = args.log_every
    model_save_path = "saved_models/"+str(args.embed_size)+"_"+str(args.hidden_size)+"_"+str(args.dropout)+"_"+args.dataset+"_"+args.model_type+"_"+args.model_save_path

    vocab_file = "data/"+args.dataset+"/"+args.vocab_file
    vocab = Vocab.load(vocab_file)

    #### INITIALIZE MODEL

    model = get_model(model_type=args.model_type,
                     embed_size=args.embed_size,
                     hidden_size=args.hidden_size,
                     vocab=vocab,
                     dropout_rate=args.dropout)

    model.train()

    tensorboard_path = f"{args.model_type}" if args.cuda else f"{args.model_type}_local"
    writer = SummaryWriter(log_dir=f"./runs/{tensorboard_path}")

    #### INITIALIZE MODEL PARAMS

    uniform_init = args.uniform_init
    if np.abs(uniform_init) > 0.:
        print('uniformly initialize parameters [-%f, +%f]' % (uniform_init, uniform_init), file=sys.stderr)
        for p in model.parameters():
            p.data.uniform_(-uniform_init, uniform_init)    #### VOCAB MASKS

    vocab_mask = torch.ones(len(vocab.vocab))
    vocab_mask[vocab.vocab['<pad>']] = 0

    #### PREPARE TRAINING
    use_cuda = args.cuda
    if isinstance(use_cuda, str):
        use_cuda = use_cuda.lower() == 'true'
    device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")
    print('use device: %s' % device, file=sys.stderr)

    model = model.to(device)
    lr = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print(f'begin Maximum Likelihood training for {args.model_type} model')

    max_epoch = args.max_epoch

    while True:
        epoch += 1

        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            train_iter += 1

            optimizer.zero_grad()

            batch_size = len(src_sents)

            example_losses = -model(src_sents, tgt_sents, device) # (batch_size,)
            batch_loss = example_losses.sum()
            loss = batch_loss / batch_size

            loss.backward()

            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()

            batch_losses_val = batch_loss.item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val

            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)
            report_tgt_words += tgt_words_num_to_predict
            cum_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_examples,
                                                                                         math.exp(report_loss / report_tgt_words),
                                                                                         cum_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         time.time() - begin_time), file=sys.stderr)

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # perform validation
            if train_iter % valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                         cum_loss / cum_examples,
                                                                                         np.exp(cum_loss / cum_tgt_words),
                                                                                         cum_examples), file=sys.stderr)

                cum_loss = cum_examples = cum_tgt_words = 0.
                valid_num += 1

                print('begin validation ...', file=sys.stderr)

                # compute dev. ppl and bleu
                dev_ppl = evaluate_ppl(model, dev_data, batch_size=128, device=device)   # dev batch size can be a bit larger
                valid_metric = -dev_ppl

                print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl), file=sys.stderr)

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    model.save(model_save_path)

                    # also save the optimizers' state
                    torch.save(optimizer.state_dict(), model_save_path + '.optim')
                elif patience < int(args.patience):
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)

                    if patience == int(args.patience):
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == int(args.max_num_trial):
                            print('early stop!', file=sys.stderr)
                            exit(0)

                        # decay lr, and restore from previously best checkpoint
                        lr = optimizer.param_groups[0]['lr'] * float(args.lr_decay)
                        print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                        # load model
                        params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                        model.load_state_dict(params['state_dict'])
                        model = model.to(device)

                        print('restore parameters of the optimizers', file=sys.stderr)
                        optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                        # reset patience
                        patience = 0

                if epoch == max_epoch:
                    print('reached maximum number of epochs!', file=sys.stderr)
                    exit(0)

def decode(args: Dict[str, str]):
    test_file = "data/"+args.dataset+"/"+args.test_file
    print("load test source sentences from [{}]".format(test_file), file=sys.stderr)
    test_data_src, test_data_tgt = read_corpus(test_file)

    test_data = list(zip(test_data_src, test_data_tgt))    
    model_load_path = "saved_models/"+str(args.embed_size)+"_"+str(args.hidden_size)+"_"+str(args.dropout)+"_"+args.dataset+"_"+args.model_type+"_"+args.model_path

    print("load model from {}".format(model_load_path), file=sys.stderr)
    model = load_model(model_load_path, args.model_type)

    use_cuda = args.cuda
    if isinstance(use_cuda, str):
        use_cuda = use_cuda.lower() == 'true'
    device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")
    model = model.to(device)

    test_ppl = evaluate_ppl(model, test_data, batch_size=128, device=device)
    test_accuracy, test_f1 = evaluate_accuracy_and_f1(model, test_data, batch_size=128, device=device)

    print('test: ppl %f, accuracy %f, f1 %f' % (test_ppl, test_accuracy, test_f1), file=sys.stderr)

def realtime(args):
    model_load_path = "saved_models/"+str(args.embed_size)+"_"+str(args.hidden_size)+"_"+str(args.dropout)+"_"+args.dataset+"_"+args.model_type+"_"+args.model_path
    print("load model from {}".format(model_load_path), file=sys.stderr)
    model = load_model(model_load_path, args.model_type)

    use_cuda = args.cuda
    if isinstance(use_cuda, str):
        use_cuda = use_cuda.lower() == 'true'
    device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.eval()
    
    print(f"Loaded {args.model_type} model. Type 'quit' to exit.")
    print("Enter a sequence of hieroglyphs (space-separated):")
    
    while True:
        try:
            user_input = input("> ").strip()
            if user_input.lower() == 'quit':
                break
                
            # Split input into tokens
            tokens = user_input.split()
            if not tokens:
                continue
                
            # Predict next token
            source = [tokens]
            prediction = model.predict_realtime(source, device)
            print(f"Next predicted hieroglyph: {prediction}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("Goodbye!")


def main():
    if args.mode == 'train':
        train(args)
    elif args.mode == 'decode':
        decode(args)
    elif args.mode == 'realtime':
        realtime(args)
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()
