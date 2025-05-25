#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from collections import Counter
from itertools import chain

def read_corpus(file_path):
    """Read corpus from file"""
    data = []
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            sent = line.strip().split()
            data.append(sent)
    return data

def build_combined_vocab(aes_train_path, ramses_train_path, output_path):
    """
    Build a combined vocabulary from AES and Ramses datasets.
    This function reads the train files from both datasets, extracts unique tokens,
    and creates a combined vocabulary file.
    """
    print("Reading AES train corpus...")
    aes_corpus = read_corpus(aes_train_path)
    print(f"AES corpus contains {len(aes_corpus)} sentences")
    
    print("Reading Ramses train corpus...")
    ramses_corpus = read_corpus(ramses_train_path)
    print(f"Ramses corpus contains {len(ramses_corpus)} sentences")
    
    # Combine and count all tokens from both datasets
    combined_tokens = list(chain(*aes_corpus + ramses_corpus))
    token_counts = Counter(combined_tokens)
    
    print(f"Total unique tokens found: {len(token_counts)}")
    
    # Create word2id dictionary
    # Add special tokens first
    word2id = {
        '<pad>': 0,
        '<unk>': 1,
        '<s>': 2,
        '</s>': 3
    }
    
    # Add all other tokens
    idx = len(word2id)
    for token in token_counts.keys():
        if token not in word2id:
            word2id[token] = idx
            idx += 1
    
    # Save the vocabulary to a JSON file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf8') as f:
        json.dump(word2id, f, ensure_ascii=False, indent=2)
    
    print(f"Combined vocabulary created with {len(word2id)} tokens")
    print(f"Saved to {output_path}")
    
    return word2id

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    aes_train_path = os.path.join(base_dir, "data", "aes", "train.txt")
    ramses_train_path = os.path.join(base_dir, "data", "ramses", "train.txt")
    output_path = os.path.join(base_dir, "data", "combined", "vocab.json")
    
    build_combined_vocab(aes_train_path, ramses_train_path, output_path)
