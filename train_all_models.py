#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to train and evaluate all model variants with the same hyperparameters
"""

import subprocess
import sys
import time
from datetime import datetime

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*50}")
    print(f"Starting: {description}")
    print(f"Command: {command}")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=False)
        end_time = time.time()
        print(f"\n✓ Completed: {description} (took {end_time - start_time:.2f} seconds)")
        return True
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        print(f"\n✗ Failed: {description} (took {end_time - start_time:.2f} seconds)")
        print(f"Error: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python train_all_models.py <dataset> [cuda] [epochs]")
        print("Example: python train_all_models.py aes True 10")
        sys.exit(1)
    
    dataset = sys.argv[1]
    cuda = sys.argv[2] if len(sys.argv) > 2 else "True"
    epochs = sys.argv[3] if len(sys.argv) > 3 else "20"
    
    models = ['lstm', 'lstm_attention', 'bilstm', 'bilstm_attention']
    
    print(f"Training all models on dataset: {dataset}")
    print(f"CUDA enabled: {cuda}")
    print(f"Max epochs: {epochs}")
    print(f"Models to train: {', '.join(models)}")
    print(f"Start time: {datetime.now()}")
    
    # Training phase
    print(f"\n{'#'*60}")
    print("TRAINING PHASE")
    print(f"{'#'*60}")
    train_results = {}
    for model_type in models:
        command = f"python main.py --dataset {dataset} --cuda {cuda} --max_epoch {epochs} --model_type {model_type}"
        success = run_command(command, f"Training {model_type} model")
        train_results[model_type] = success
    
    # Testing phase
    print(f"\n{'#'*60}")
    print("TESTING PHASE")
    print(f"{'#'*60}")
    
    test_results = {}
    for model_type in models:
        if train_results[model_type]:  # Only test if training succeeded
            command = f"python main.py --dataset {dataset} --cuda {cuda} --mode decode --model_type {model_type}"
            success = run_command(command, f"Testing {model_type} model")
            test_results[model_type] = success
        else:
            print(f"\nSkipping test for {model_type} - training failed")
            test_results[model_type] = False
    
    # Multi-shot evaluation phase
    print(f"\n{'#'*60}")
    print("MULTI-SHOT EVALUATION PHASE")
    print(f"{'#'*60}")
    multishot_results = {}
    for model_type in models:
        if train_results[model_type]:  # Only evaluate if training succeeded
            command = f"python multi.py --dataset {dataset} --cuda {cuda} --model_type {model_type}"
            success = run_command(command, f"Multi-shot evaluation for {model_type} model")
            multishot_results[model_type] = success
        else:
            print(f"\nSkipping multi-shot evaluation for {model_type} - training failed")
            multishot_results[model_type] = False
    
    # Summary
    print(f"\n{'#'*60}")
    print("SUMMARY")
    print(f"{'#'*60}")
    print(f"End time: {datetime.now()}")
    
    print("\nTraining Results:")
    for model_type in models:
        status = "✓ SUCCESS" if train_results[model_type] else "✗ FAILED"
        print(f"  {model_type:20} {status}")
    
    print("\nTesting Results:")
    for model_type in models:
        if model_type in test_results:
            status = "✓ SUCCESS" if test_results[model_type] else "✗ FAILED"
            print(f"  {model_type:20} {status}")
    
    print("\nMulti-shot Evaluation Results:")
    for model_type in models:
        if model_type in multishot_results:
            status = "✓ SUCCESS" if multishot_results[model_type] else "✗ FAILED"
            print(f"  {model_type:20} {status}")
    
    # Check if all succeeded
    all_train_success = all(train_results.values())
    all_test_success = all(test_results.values())
    all_multishot_success = all(multishot_results.values())
    
    if all_train_success and all_test_success and all_multishot_success:
        print(f"\n🎉 All models trained and evaluated successfully!")
    else:
        print(f"\n⚠️  Some models failed. Check the logs above.")

if __name__ == "__main__":
    main()
