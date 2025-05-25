#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple test script to verify all models can be loaded and run
"""

import sys
import torch
import traceback
from models import get_model, load_model
from vocab import Vocab, VocabEntry

def create_test_vocab():
    """Create a simple test vocabulary"""
    vocab_data = {
        '<pad>': 0,
        '<unk>': 1,
        '<s>': 2,
        '</s>': 3,
        'test1': 4,
        'test2': 5,
        'test3': 6
    }
    
    vocab_entry = VocabEntry(vocab_data)
    vocab = Vocab(vocab_entry)
    return vocab

def test_models():
    """Test all model types"""
    print("Testing all model architectures...")
    
    vocab = create_test_vocab()
    model_types = ['lstm', 'lstm_attention', 'bilstm', 'bilstm_attention']
    embed_size = 64
    hidden_size = 64
    dropout_rate = 0.1
    device = torch.device('cpu')
    
    test_input = [['test1', 'test2'], ['test3', 'test1']]
    test_target = [['test2', 'test3'], ['test1', 'test2']]
    
    results = {}
    
    for model_type in model_types:
        print(f"\nTesting {model_type}...")
        try:
            # Create model
            model = get_model(
                model_type=model_type,
                embed_size=embed_size,
                hidden_size=hidden_size,
                vocab=vocab,
                dropout_rate=dropout_rate
            )
            
            # Test forward pass
            scores = model(test_input, test_target, device)
            print(f"  Forward pass successful: scores shape {scores.shape}")
            
            # Test prediction
            if model_type in ['lstm', 'bilstm']:
                pred, masks, padded, lens = model.predict(test_input, test_target, device)
                print(f"  Prediction successful: shape {pred.shape}")
            
            # Test real-time prediction
            pred = model.predict_realtime(test_input, device)
            print(f"  Real-time prediction: {pred}")
            
            results[model_type] = "✓ SUCCESS"
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            traceback.print_exc()
            results[model_type] = "✗ FAILED"
    
    # Print summary
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    for model_type, result in results.items():
        print(f"{model_type:20} {result}")
    
    if all(r == "✓ SUCCESS" for r in results.values()):
        print("\nAll models passed tests! 🎉")
        return True
    else:
        print("\nSome models failed. Check the logs above.")
        return False

if __name__ == "__main__":
    success = test_models()
    sys.exit(0 if success else 1)
