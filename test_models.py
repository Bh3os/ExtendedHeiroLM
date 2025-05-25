#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script to verify all model variants can be instantiated and run basic operations
"""

import sys
import torch
import traceback
from models import get_model, HieroLM, HieroLMAttention, BiHieroLM, BiHieroLMAttention
from vocab import Vocab

def test_model_creation():
    """Test that all model types can be created successfully"""
    print("Testing model creation...")
    
    # Create a dummy vocab for testing
    vocab_data = {
        '<pad>': 0,
        '<unk>': 1,
        '<s>': 2,
        '</s>': 3,
        'hieroglyph1': 4,
        'hieroglyph2': 5,
        'hieroglyph3': 6
    }
    
    from vocab import VocabEntry
    vocab_entry = VocabEntry(vocab_data)
    vocab = Vocab(vocab_entry)
    
    models_to_test = ['lstm', 'lstm_attention', 'bilstm', 'bilstm_attention']
    embed_size = 128
    hidden_size = 64
    dropout_rate = 0.1
    
    results = {}
    
    for model_type in models_to_test:
        try:
            print(f"  Creating {model_type}...")
            model = get_model(model_type, embed_size, hidden_size, vocab, dropout_rate)
            
            # Test model parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"    ✓ {model_type} created successfully")
            print(f"      Total parameters: {total_params:,}")
            print(f"      Trainable parameters: {trainable_params:,}")
            
            results[model_type] = {
                'status': 'success',
                'total_params': total_params,
                'trainable_params': trainable_params
            }
            
        except Exception as e:
            print(f"    ✗ {model_type} failed: {e}")
            results[model_type] = {
                'status': 'failed',
                'error': str(e)
            }
    
    return results, vocab

def test_model_forward_pass(model, model_type, vocab, device):
    """Test forward pass with dummy data"""
    try:
        # Create dummy input data
        source = [['hieroglyph1', 'hieroglyph2'], ['hieroglyph3']]
        target = [['hieroglyph2', '</s>'], ['</s>']]
        
        model = model.to(device)
        model.eval()
        
        with torch.no_grad():
            # Test forward pass
            scores = model(source, target, device)
            
            # Test prediction
            predictions, masks, target_padded, source_lengths = model.predict(source, target, device)
            
            # Test real-time prediction
            realtime_pred = model.predict_realtime([['hieroglyph1']], device)
            
        print(f"    ✓ {model_type} forward pass successful")
        print(f"      Scores shape: {scores.shape}")
        print(f"      Predictions shape: {predictions.shape}")
        print(f"      Real-time prediction: {realtime_pred}")
        
        return True
        
    except Exception as e:
        print(f"    ✗ {model_type} forward pass failed: {e}")
        print(f"      Error details: {traceback.format_exc()}")
        return False

def test_model_save_load(model, model_type, vocab, device):
    """Test model saving and loading"""
    try:
        # Save model
        save_path = f"test_{model_type}_model.bin"
        model.save(save_path)
        print(f"    ✓ {model_type} saved successfully")
        
        # Load model
        from models import load_model
        loaded_model = load_model(save_path, model_type)
        loaded_model = loaded_model.to(device)
        
        print(f"    ✓ {model_type} loaded successfully")
        
        # Test that loaded model works
        source = [['hieroglyph1']]
        target = [['</s>']]
        
        with torch.no_grad():
            original_scores = model(source, target, device)
            loaded_scores = loaded_model(source, target, device)
            
        # Check if scores are similar (should be identical)
        if torch.allclose(original_scores, loaded_scores, atol=1e-6):
            print(f"    ✓ {model_type} save/load consistency verified")
            return True
        else:
            print(f"    ⚠ {model_type} save/load scores differ")
            return False
            
    except Exception as e:
        print(f"    ✗ {model_type} save/load failed: {e}")
        return False

def main():
    print("HieroLM Multi-Model Test Suite")
    print("=" * 50)
    
    # Test model creation
    creation_results, vocab = test_model_creation()
    
    print("\n" + "=" * 50)
    print("Testing forward passes...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    forward_results = {}
    save_load_results = {}
    
    for model_type, result in creation_results.items():
        if result['status'] == 'success':
            print(f"\n  Testing {model_type}...")
            
            # Create model again for testing
            model = get_model(model_type, 128, 64, vocab, 0.1)
            
            # Test forward pass
            forward_success = test_model_forward_pass(model, model_type, vocab, device)
            forward_results[model_type] = forward_success
            
            # Test save/load if forward pass succeeded
            if forward_success:
                save_load_success = test_model_save_load(model, model_type, vocab, device)
                save_load_results[model_type] = save_load_success
            else:
                save_load_results[model_type] = False
        else:
            forward_results[model_type] = False
            save_load_results[model_type] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    print(f"{'Model Type':<20} {'Creation':<10} {'Forward':<10} {'Save/Load':<10}")
    print("-" * 60)
    
    all_passed = True
    for model_type in ['lstm', 'lstm_attention', 'bilstm', 'bilstm_attention']:
        creation = "✓" if creation_results.get(model_type, {}).get('status') == 'success' else "✗"
        forward = "✓" if forward_results.get(model_type, False) else "✗"
        save_load = "✓" if save_load_results.get(model_type, False) else "✗"
        
        print(f"{model_type:<20} {creation:<10} {forward:<10} {save_load:<10}")
        
        if not (creation == "✓" and forward == "✓" and save_load == "✓"):
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! All model variants are working correctly.")
    else:
        print("⚠️  Some tests failed. Check the detailed output above.")
    
    # Cleanup test files
    import os
    for model_type in ['lstm', 'lstm_attention', 'bilstm', 'bilstm_attention']:
        test_file = f"test_{model_type}_model.bin"
        if os.path.exists(test_file):
            os.remove(test_file)
    
    print("✓ Cleanup completed.")

if __name__ == "__main__":
    main()
