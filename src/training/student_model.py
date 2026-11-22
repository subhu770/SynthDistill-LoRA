"""Lightweight, CPU-runnable Causal Transformer Student Model and ChatML Tokenizer."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.api.schemas import SyntheticSample


class ChatMLTokenizer:
    """Zero-dependency deterministic byte-level tokenizer with ChatML special tokens."""

    SPECIAL_TOKENS: Dict[str, int] = {
        "<pad>": 0,
        "<unk>": 1,
        "<bos>": 2,
        "<eos>": 3,
        "<|im_start|>": 4,
        "<|im_end|>": 5,
        "<|system|>": 6,
        "<|user|>": 7,
        "<|assistant|>": 8,
    }
    BYTE_OFFSET = 10  # Bytes 0-255 map to indices 10..265
    VOCAB_SIZE = 266  # 10 special tokens + 256 byte values

    def __init__(self) -> None:
        self.pad_token_id = self.SPECIAL_TOKENS["<pad>"]
        self.eos_token_id = self.SPECIAL_TOKENS["<eos>"]
        self.bos_token_id = self.SPECIAL_TOKENS["<bos>"]
        self.id_to_special = {v: k for k, v in self.SPECIAL_TOKENS.items()}

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encode string into integer token IDs using UTF-8 byte encoding."""
        tokens: List[int] = []
        if add_special_tokens:
            tokens.append(self.bos_token_id)

        raw_bytes = text.encode("utf-8")
        for b in raw_bytes:
            tokens.append(b + self.BYTE_OFFSET)

        if add_special_tokens:
            tokens.append(self.eos_token_id)
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Decode integer token IDs back to string."""
        byte_list: List[int] = []
        for tid in token_ids:
            if tid in self.id_to_special:
                continue
            if self.BYTE_OFFSET <= tid < self.VOCAB_SIZE:
                byte_list.append(tid - self.BYTE_OFFSET)
        return bytes(byte_list).decode("utf-8", errors="replace")

    def encode_chatml(
        self,
        sample: SyntheticSample,
        max_seq_len: int = 128,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Format and tokenize a SyntheticSample for Causal LM training.

        Labels for the prompt portion are masked with -100 so cross-entropy loss
        is only computed on the assistant response.

        Returns:
            Tuple of (input_ids, labels) tensors of shape (max_seq_len,)
        """
        # Format prompt
        system_text = sample.get_system_prompt() or "You are a helpful assistant."
        prompt_text = sample.get_prompt()
        response_text = sample.get_response()

        formatted_prompt = (
            f"<|im_start|>system\n{system_text}<|im_end|>\n"
            f"<|im_start|>user\n{prompt_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        formatted_response = f"{response_text}<|im_end|>"

        prompt_ids = self.encode(formatted_prompt)
        response_ids = self.encode(formatted_response)

        all_ids = prompt_ids + response_ids
        if len(all_ids) > max_seq_len:
            # Truncate prompt from start if too long, keep response
            excess = len(all_ids) - max_seq_len
            if excess < len(prompt_ids):
                prompt_ids = prompt_ids[excess:]
            else:
                prompt_ids = prompt_ids[:10]
            all_ids = (prompt_ids + response_ids)[:max_seq_len]

        # Build labels: -100 for prompt tokens, token_ids for response tokens
        labels = [-100] * len(prompt_ids) + all_ids[len(prompt_ids) :]

        # Pad to max_seq_len
        pad_len = max_seq_len - len(all_ids)
        if pad_len > 0:
            all_ids = all_ids + [self.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len

        return (
            torch.tensor(all_ids[:max_seq_len], dtype=torch.long),
            torch.tensor(labels[:max_seq_len], dtype=torch.long),
        )


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with explicit projection layers for LoRA adaptation."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Explicit linear projections to facilitate clean target module substitution
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # Project and reshape: (B, n_heads, T, head_dim)
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        scores = scores.masked_fill(~causal_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = attn @ v  # (B, n_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Standard Pre-LN Transformer decoder block."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln_2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.fc2(self.act(self.fc1(self.ln_2(x))))
        return x


class MiniCausalLM(nn.Module):
    """Lightweight Causal Transformer Student Model runnable on CPU in seconds."""

    def __init__(
        self,
        vocab_size: int = ChatMLTokenizer.VOCAB_SIZE,
        d_model: int = 64,
        n_heads: int = 2,
        n_layers: int = 2,
        max_seq_len: int = 128,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(0.1)

        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying or parameter initialization
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass computing causal language model logits and loss.

        Args:
            input_ids: Tensor of token ids (batch_size, seq_len)
            labels: Optional tensor of target labels (batch_size, seq_len) with -100 for ignored positions

        Returns:
            Tuple of (logits, loss)
        """
        B, T = input_ids.size()
        if T > self.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}")

        positions = torch.arange(0, T, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(positions)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if labels is not None:
            # Shift tokens for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 0.7,
    ) -> torch.Tensor:
        """Autoregressively generate next token IDs."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = (
                input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len :]
            )
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)
        return input_ids
