﻿import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--max_epoch', default=50, type=int, help='number of epochs')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--dataset', default="aes", type=str, help='path')
    parser.add_argument('--train_file', default="train.txt", type=str, help='path')
    parser.add_argument('--dev_file', default="val.txt", type=str, help='path')
    parser.add_argument('--test_file', default="test.txt", type=str, help='path')
    parser.add_argument('--train_batch_size', default=32, type=int, help='batch')
    parser.add_argument('--clip_grad', default=5.0, type=float, help='clip')
    parser.add_argument('--valid_niter', default=200, type=int, help='valid')
    parser.add_argument('--log_every', default=10, type=int, help='log')
    parser.add_argument('--model_save_path', default="model.bin", type=str, help='path')
    parser.add_argument('--model_path', default="model.bin", type=str, help='path')
    parser.add_argument('--vocab_file', default="vocab.json", type=str, help='path')
    parser.add_argument('--dropout', default=0, type=float, help='rate for edge dropout')
    parser.add_argument('--embed_size', default=1024, type=int, help='embed_size')
    parser.add_argument('--hidden_size', default=1024, type=int, help='hidden_size')
    parser.add_argument('--uniform_init', default=0.1, type=float, help='init')
    parser.add_argument('--cuda', default=False, type=bool, help='the gpu to use')
    parser.add_argument('--patience', default=5, type=int, help='patience')
    parser.add_argument('--max_num_trial', default=5, type=int, help='trial')
    parser.add_argument('--lr_decay', default=0.5, type=float, help='lr_decay')
    parser.add_argument('--mode', default="train", type=str, help='mode')
    parser.add_argument('--model_type', default="lstm", type=str, 
                       choices=['lstm', 'lstm_attention', 'bilstm', 'bilstm_attention'],
                       help='Type of model to use: lstm, lstm_attention, bilstm, bilstm_attention')
    return parser.parse_args()

args = parse_args()
