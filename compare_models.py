#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to compare performance of different model variants
"""

import os
import sys
import torch
import json
from models import load_model
from utils import read_corpus, batch_iter
from vocab import Vocab
import numpy as np
from tqdm import tqdm
import time
from parse import args

def evaluate_model_performance(model_type, dataset, cuda=True):
    """Evaluate a specific model and return metrics"""
    
    # Load test data
    test_file = f"data/{dataset}/test.txt"
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return None
    
    test_data_src, test_data_tgt = read_corpus(test_file)
    test_data = list(zip(test_data_src, test_data_tgt))
    
    # Load model
    model_path = f"saved_models/1024_1024_0_{dataset}_{model_type}_model.bin"
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None
    
    try:
        model = load_model(model_path, model_type)
        device = torch.device("cuda:0" if cuda and torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        # Evaluate perplexity
        start_time = time.time()
        cum_loss = 0.
        cum_tgt_words = 0.
        
        with torch.no_grad():
            for src_sents, tgt_sents in tqdm(batch_iter(test_data, 128), desc=f"Evaluating {model_type}"):
                loss = -model(src_sents, tgt_sents, device).sum()
                cum_loss += loss.item()
                tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)
                cum_tgt_words += tgt_word_num_to_predict
        
        ppl = np.exp(cum_loss / cum_tgt_words)
        eval_time = time.time() - start_time
        
        # Evaluate accuracy
        all_predictions = []
        all_targets = []
        
        start_time = time.time()
        with torch.no_grad():
            for src_sents, tgt_sents in tqdm(batch_iter(test_data, 128), desc=f"Computing accuracy for {model_type}"):
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
        
        accuracy = sum(p == t for p, t in zip(all_predictions, all_targets)) / len(all_predictions) if all_predictions else 0
        accuracy_time = time.time() - start_time
        
        # Count model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'model_type': model_type,
            'perplexity': float(ppl),
            'accuracy': float(accuracy),
            'total_params': total_params,
            'trainable_params': trainable_params,
            'eval_time': eval_time,
            'accuracy_time': accuracy_time,
            'status': 'success'
        }
        
    except Exception as e:
        print(f"Error evaluating {model_type}: {e}")
        return {
            'model_type': model_type,
            'status': 'error',
            'error': str(e)
        }

def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_models.py <dataset> [cuda]")
        print("Example: python compare_models.py aes True")
        sys.exit(1)
    
    dataset = sys.argv[1]
    cuda = sys.argv[2].lower() == 'true' if len(sys.argv) > 2 else True
    
    models = ['lstm', 'lstm_attention', 'bilstm', 'bilstm_attention']
    
    print(f"Comparing models on dataset: {dataset}")
    print(f"CUDA enabled: {cuda}")
    print(f"Models to compare: {', '.join(models)}")
    print(f"{'='*80}")
    
    results = []
    
    for model_type in models:
        print(f"\nEvaluating {model_type}...")
        result = evaluate_model_performance(model_type, dataset, cuda)
        if result:
            results.append(result)
    
    # Display results
    print(f"\n{'='*80}")
    print("COMPARISON RESULTS")
    print(f"{'='*80}")
    
    if not results:
        print("No models could be evaluated.")
        return
    
    # Filter successful results
    success_results = [r for r in results if r['status'] == 'success']
    
    if not success_results:
        print("No models evaluated successfully.")
        return
    
    # Sort by perplexity (lower is better)
    success_results.sort(key=lambda x: x['perplexity'])
    
    print(f"\n{'Model Type':<20} {'Perplexity':<12} {'Accuracy':<10} {'Parameters':<12} {'Eval Time':<10}")
    print("-" * 70)
    
    for result in success_results:
        print(f"{result['model_type']:<20} {result['perplexity']:<12.4f} {result['accuracy']:<10.4f} "
              f"{result['total_params']:<12,} {result['eval_time']:<10.2f}s")
    
    # Best model
    best_model = success_results[0]
    print(f"\n🏆 Best model (lowest perplexity): {best_model['model_type']}")
    print(f"   Perplexity: {best_model['perplexity']:.4f}")
    print(f"   Accuracy: {best_model['accuracy']:.4f}")
    print(f"   Parameters: {best_model['total_params']:,}")
    
    # Best accuracy
    best_accuracy = max(success_results, key=lambda x: x['accuracy'])
    if best_accuracy['model_type'] != best_model['model_type']:
        print(f"\n🎯 Best accuracy: {best_accuracy['model_type']}")
        print(f"   Accuracy: {best_accuracy['accuracy']:.4f}")
        print(f"   Perplexity: {best_accuracy['perplexity']:.4f}")
    
    # Model complexity comparison
    print(f"\n📊 Model Complexity:")
    for result in success_results:
        print(f"   {result['model_type']:<20} {result['total_params']:>12,} parameters")
    
    # Save results to JSON
    output_file = f"{dataset}_model_comparison.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {output_file}")
    
    # Show failed models
    failed_results = [r for r in results if r['status'] == 'error']
    if failed_results:
        print(f"\n❌ Failed evaluations:")
        for result in failed_results:
            print(f"   {result['model_type']}: {result['error']}")

if __name__ == "__main__":
    main()
