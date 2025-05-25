#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import math

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


class AttentionModule(nn.Module):
    """Attention mechanism for LSTM models"""
    def __init__(self, hidden_size):
        super(AttentionModule, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        
    def forward(self, hidden, encoder_outputs, source_lengths):
        """
        hidden: (batch_size, hidden_size) - current hidden state
        encoder_outputs: (src_len, batch_size, hidden_size) - all encoder outputs
        source_lengths: list of source sequence lengths
        """
        batch_size = encoder_outputs.size(1)
        src_len = encoder_outputs.size(0)
        
        # Repeat hidden state for all encoder outputs
        hidden = hidden.unsqueeze(0).repeat(src_len, 1, 1)  # (src_len, batch_size, hidden_size)
        
        # Calculate attention weights
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # (src_len, batch_size, hidden_size)
        energy = energy.permute(1, 0, 2)  # (batch_size, src_len, hidden_size)
        
        v = self.v.repeat(batch_size, 1).unsqueeze(1)  # (batch_size, 1, hidden_size)
        attention_weights = torch.bmm(v, energy.transpose(1, 2)).squeeze(1)  # (batch_size, src_len)
        
        # Apply padding mask
        mask = torch.zeros(batch_size, src_len, device=encoder_outputs.device)
        for i, length in enumerate(source_lengths):
            mask[i, :length] = 1
        
        attention_weights = attention_weights.masked_fill(mask == 0, -1e10)
        attention_weights = F.softmax(attention_weights, dim=1)  # (batch_size, src_len)
        
        # Apply attention weights to encoder outputs
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # (batch_size, src_len, hidden_size)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch_size, hidden_size)
        
        return context, attention_weights


class BaseHieroLM(nn.Module):
    """Base class for all HieroLM variants"""
    def __init__(self, embed_size, hidden_size, vocab, dropout_rate):
        super(BaseHieroLM, self).__init__()
        src_pad_token_idx = vocab.vocab['<pad>']
        self.embed_size = embed_size
        self.model_embeddings = nn.Embedding(len(vocab.vocab), embed_size, padding_idx=src_pad_token_idx)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab
        self.dropout = nn.Dropout(dropout_rate)
        
        # This will be defined in child classes
        self.encoder = None
        self.target_vocab_projection = None
        
    def forward(self, source: List[List[str]], target: List[List[str]], device) -> torch.Tensor:
        # Compute sentence lengths
        source_lengths = [len(s) for s in source]

        # Convert list of lists into tensors
        source_padded = self.vocab.vocab.to_input_tensor(source, device=device)  # Tensor: (src_len, b)
        target_padded = self.vocab.vocab.to_input_tensor(target, device=device)  # Tensor: (tgt_len, b)

        enc_hiddens = self.encode(source_padded, source_lengths)

        P = F.log_softmax(self.target_vocab_projection(enc_hiddens), dim=-1)

        # Zero out, probabilities for which we have nothing in the target text
        target_masks = (target_padded != self.vocab.vocab['<pad>']).float()

        # Compute log probability of generating true target words
        target_gold_words_log_prob = torch.gather(P, index=target_padded.unsqueeze(-1), dim=-1).squeeze(
            -1) * target_masks
        scores = target_gold_words_log_prob.sum(dim=0)
        return scores

    def predict(self, source: List[List[str]], target: List[List[str]], device) -> torch.Tensor:
        source_lengths = [len(s) for s in source]
        source_padded = self.vocab.vocab.to_input_tensor(source, device=device)  # Tensor: (src_len, b)
        target_padded = self.vocab.vocab.to_input_tensor(target, device=device)  # Tensor: (tgt_len, b)
        enc_hiddens = self.encode(source_padded, source_lengths)

        P = F.log_softmax(self.target_vocab_projection(enc_hiddens), dim=-1)

        # Zero out, probabilities for which we have nothing in the target text
        target_masks = (target_padded != self.vocab.vocab['<pad>']).float()

        predictions = torch.argmax(P, dim=-1) * target_masks

        return predictions, target_masks, target_padded, source_lengths
    
    def predict_realtime(self, source: List[List[str]], device) -> str:
        source_lengths = [len(s) for s in source]
        source_padded = self.vocab.vocab.to_input_tensor(source, device=device)  # Tensor: (src_len, b)
        enc_hiddens = self.encode(source_padded, source_lengths)

        P = F.log_softmax(self.target_vocab_projection(enc_hiddens), dim=-1)

        prediction_idx = torch.argmax(P, dim=-1)[-1][0].cpu().item()
        prediction = self.vocab.vocab.id2word[prediction_idx]

        return prediction

    @property
    def device(self) -> torch.device:
        return self.model_embeddings.weight.device

    def save(self, path: str):
        print('save model parameters to [%s]' % path, file=sys.stderr)
        params = {
            'args': dict(embed_size=self.embed_size, hidden_size=self.hidden_size,
                        dropout_rate=self.dropout_rate),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(params, path)


class HieroLM(BaseHieroLM):
    """Original LSTM model"""
    def __init__(self, embed_size, hidden_size, vocab, dropout_rate):
        super(HieroLM, self).__init__(embed_size, hidden_size, vocab, dropout_rate)
        
        self.encoder = nn.LSTM(embed_size, hidden_size, bias=True, bidirectional=False, dropout=dropout_rate if dropout_rate > 0 else 0)
        self.target_vocab_projection = nn.Linear(hidden_size, len(vocab.vocab), bias=False)

    def encode(self, source_padded: torch.Tensor, source_lengths: List[int]) -> torch.Tensor:
        X = self.model_embeddings(source_padded)
        X = self.dropout(X)
        enc_hiddens, (last_hidden, last_cell) = self.encoder(pack_padded_sequence(X, source_lengths))
        enc_hiddens = pad_packed_sequence(enc_hiddens)[0]
        return enc_hiddens

    @staticmethod
    def load(model_path: str):
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = HieroLM(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])
        return model


class HieroLMAttention(BaseHieroLM):
    """LSTM with attention mechanism"""
    def __init__(self, embed_size, hidden_size, vocab, dropout_rate):
        super(HieroLMAttention, self).__init__(embed_size, hidden_size, vocab, dropout_rate)
        
        self.encoder = nn.LSTM(embed_size, hidden_size, bias=True, bidirectional=False, dropout=dropout_rate if dropout_rate > 0 else 0)
        self.attention = AttentionModule(hidden_size)
        self.target_vocab_projection = nn.Linear(hidden_size * 2, len(vocab.vocab), bias=False)  # *2 for context + hidden

    def encode(self, source_padded: torch.Tensor, source_lengths: List[int]) -> torch.Tensor:
        X = self.model_embeddings(source_padded)
        X = self.dropout(X)
        enc_hiddens, (last_hidden, last_cell) = self.encoder(pack_padded_sequence(X, source_lengths))
        enc_hiddens = pad_packed_sequence(enc_hiddens)[0]
        
        # Apply attention for each time step
        seq_len, batch_size, hidden_size = enc_hiddens.shape
        attended_outputs = []
        
        for t in range(seq_len):
            current_hidden = enc_hiddens[t]  # (batch_size, hidden_size)
            context, _ = self.attention(current_hidden, enc_hiddens, source_lengths)
            # Concatenate hidden state with context
            attended_output = torch.cat([current_hidden, context], dim=1)  # (batch_size, hidden_size * 2)
            attended_outputs.append(attended_output)
        
        attended_outputs = torch.stack(attended_outputs, dim=0)  # (seq_len, batch_size, hidden_size * 2)
        return attended_outputs

    @staticmethod
    def load(model_path: str):
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = HieroLMAttention(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])
        return model


class BiHieroLM(BaseHieroLM):
    """Bidirectional LSTM model"""
    def __init__(self, embed_size, hidden_size, vocab, dropout_rate):
        super(BiHieroLM, self).__init__(embed_size, hidden_size, vocab, dropout_rate)
        
        self.encoder = nn.LSTM(embed_size, hidden_size, bias=True, bidirectional=True, dropout=dropout_rate if dropout_rate > 0 else 0)
        self.target_vocab_projection = nn.Linear(hidden_size * 2, len(vocab.vocab), bias=False)  # *2 for bidirectional

    def encode(self, source_padded: torch.Tensor, source_lengths: List[int]) -> torch.Tensor:
        X = self.model_embeddings(source_padded)
        X = self.dropout(X)
        enc_hiddens, (last_hidden, last_cell) = self.encoder(pack_padded_sequence(X, source_lengths))
        enc_hiddens = pad_packed_sequence(enc_hiddens)[0]
        return enc_hiddens

    @staticmethod
    def load(model_path: str):
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = BiHieroLM(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])
        return model


class BiHieroLMAttention(BaseHieroLM):
    """Bidirectional LSTM with attention mechanism"""
    def __init__(self, embed_size, hidden_size, vocab, dropout_rate):
        super(BiHieroLMAttention, self).__init__(embed_size, hidden_size, vocab, dropout_rate)
        
        self.encoder = nn.LSTM(embed_size, hidden_size, bias=True, bidirectional=True, dropout=dropout_rate if dropout_rate > 0 else 0)
        self.attention = AttentionModule(hidden_size * 2)  # *2 for bidirectional
        self.target_vocab_projection = nn.Linear(hidden_size * 4, len(vocab.vocab), bias=False)  # *4 for bidirectional + context

    def encode(self, source_padded: torch.Tensor, source_lengths: List[int]) -> torch.Tensor:
        X = self.model_embeddings(source_padded)
        X = self.dropout(X)
        enc_hiddens, (last_hidden, last_cell) = self.encoder(pack_padded_sequence(X, source_lengths))
        enc_hiddens = pad_packed_sequence(enc_hiddens)[0]
        
        # Apply attention for each time step
        seq_len, batch_size, hidden_size = enc_hiddens.shape
        attended_outputs = []
        
        for t in range(seq_len):
            current_hidden = enc_hiddens[t]  # (batch_size, hidden_size * 2)
            context, _ = self.attention(current_hidden, enc_hiddens, source_lengths)
            # Concatenate hidden state with context
            attended_output = torch.cat([current_hidden, context], dim=1)  # (batch_size, hidden_size * 4)
            attended_outputs.append(attended_output)
        
        attended_outputs = torch.stack(attended_outputs, dim=0)  # (seq_len, batch_size, hidden_size * 4)
        return attended_outputs

    @staticmethod
    def load(model_path: str):
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = BiHieroLMAttention(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])
        return model


# Model factory function
def get_model(model_type: str, embed_size: int, hidden_size: int, vocab, dropout_rate: float):
    """Factory function to create different model variants"""
    model_classes = {
        'lstm': HieroLM,
        'lstm_attention': HieroLMAttention,
        'bilstm': BiHieroLM,
        'bilstm_attention': BiHieroLMAttention
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}. Available types: {list(model_classes.keys())}")
    
    return model_classes[model_type](embed_size, hidden_size, vocab, dropout_rate)


def load_model(model_path: str, model_type: str):
    """Factory function to load different model variants"""
    model_classes = {
        'lstm': HieroLM,
        'lstm_attention': HieroLMAttention,
        'bilstm': BiHieroLM,
        'bilstm_attention': BiHieroLMAttention
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}. Available types: {list(model_classes.keys())}")
    
    return model_classes[model_type].load(model_path)
