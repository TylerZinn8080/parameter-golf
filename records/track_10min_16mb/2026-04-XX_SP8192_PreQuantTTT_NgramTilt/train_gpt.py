from __future__ import annotations

import base64
import copy
import glob
import io
import json
import lzma
import math
import os
import queue
import random
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import zstandard  # type: ignore
except Exception:
    zstandard = None

# Compression priority must be LZMA-9 primary (smaller artifacts on this distro).
_COMPRESSOR = "lzma"

try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func  # type: ignore
except Exception:
    flash_attn_3_func = None


# ── TOKENIZER / DATA ──────────────────────────────────────────────────
VOCAB_SIZE = int(os.environ.get("VOCAB_SIZE", 8192))
DATA_PATH = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp8192/")
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_8192_bpe.model")

# ── ARCHITECTURE ──────────────────────────────────────────────────────
NUM_LAYERS = int(os.environ.get("NUM_LAYERS", 11))
D_MODEL = int(os.environ.get("D_MODEL", 512))
NUM_HEADS = int(os.environ.get("NUM_HEADS", 8))
NUM_KV_HEADS = int(os.environ.get("NUM_KV_HEADS", 4))
MLP_MULT = float(os.environ.get("MLP_MULT", 4.0))
INIT_STD = float(os.environ.get("INIT_STD", 0.005))
LOGIT_SOFTCAP = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
QK_GAIN = float(os.environ.get("QK_GAIN", 5.0))

# ── DEPTH RECURRENCE ──────────────────────────────────────────────────
NUM_LOOPS = int(os.environ.get("NUM_LOOPS", 3))
# Loop layers 4-5, NUM_LOOPS extra passes:
# pattern [0,1,2,3,4,5,...4,5,6,7,8,9,10] (4-5 appear NUM_LOOPS+1 times)
ENABLE_LOOPING_AT = float(os.environ.get("ENABLE_LOOPING_AT", 0.35))

# ── ATTENTION ─────────────────────────────────────────────────────────
ROPE_DIMS = int(os.environ.get("ROPE_DIMS", 16))  # partial RoPE
ROPE_BASE = int(os.environ.get("ROPE_BASE", 10000))
SEQ_LEN = int(os.environ.get("SEQ_LEN", 2048))

# ── REGULARIZATION ───────────────────────────────────────────────────
LN_SCALE = bool(int(os.environ.get("LN_SCALE", 1)))
SMEARGATE = bool(int(os.environ.get("SMEARGATE", 1)))
VE_ENABLED = bool(int(os.environ.get("VE_ENABLED", 1)))
VE_DIM = int(os.environ.get("VE_DIM", 128))
VE_LAYERS = [int(x) for x in os.environ.get("VE_LAYERS", "9,10").split(",")]

# ── BIGRAM HASH ───────────────────────────────────────────────────────
BIGRAM_VOCAB_SIZE = int(os.environ.get("BIGRAM_VOCAB_SIZE", 3072))
BIGRAM_DIM = int(os.environ.get("BIGRAM_DIM", 112))

# ── U-NET SKIPS ───────────────────────────────────────────────────────
UNET_SKIPS = bool(int(os.environ.get("UNET_SKIPS", 1)))

# ── PARALLEL RESIDUALS ────────────────────────────────────────────────
PARALLEL_RESIDUAL_START = int(os.environ.get("PARALLEL_RESIDUAL_START", 7))

# ── XSA (Cross-Sequence Attention) ───────────────────────────────────
XSA_ALL_LAYERS = bool(int(os.environ.get("XSA_ALL_LAYERS", 1)))

# ── OPTIMIZER ─────────────────────────────────────────────────────────
MATRIX_LR = float(os.environ.get("MATRIX_LR", 0.025))
SCALAR_LR = float(os.environ.get("SCALAR_LR", 0.025))
TIED_EMBED_LR = float(os.environ.get("TIED_EMBED_LR", 0.035))
MUON_WD = float(os.environ.get("MUON_WD", 0.085))
ADAM_WD = float(os.environ.get("ADAM_WD", 0.085))
MUON_MOMENTUM = float(os.environ.get("MUON_MOMENTUM", 0.99))
MUON_MOM_WARMUP_START = float(os.environ.get("MUON_MOM_WARMUP_START", 0.92))
MUON_MOM_WARMUP_STEPS = int(os.environ.get("MUON_MOM_WARMUP_STEPS", 1500))

# ── WEIGHT AVERAGING ─────────────────────────────────────────────────
EMA_ENABLED = bool(int(os.environ.get("EMA_ENABLED", 1)))
EMA_DECAY = float(os.environ.get("EMA_DECAY", 0.997))
SWA_ENABLED = bool(int(os.environ.get("SWA_ENABLED", 1)))
SWA_EVERY = int(os.environ.get("SWA_EVERY", 50))

# ── TRAINING SCHEDULE ─────────────────────────────────────────────────
WARMDOWN_ITERS = int(os.environ.get("WARMDOWN_ITERS", 4000))
WARMUP_ITERS = int(os.environ.get("WARMUP_ITERS", 300))
MAX_WALLCLOCK_SECONDS = int(os.environ.get("MAX_WALLCLOCK_SECONDS", 600))
TRAIN_BATCH_TOKENS = int(os.environ.get("TRAIN_BATCH_TOKENS", 524288))
GRAD_CLIP = float(os.environ.get("GRAD_CLIP", 1.0))

# ── QUANTIZATION: LATE QAT ────────────────────────────────────────────
LATE_QAT = bool(int(os.environ.get("LATE_QAT", 1)))
LATE_QAT_THRESHOLD = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))

# ── QUANTIZATION: GPTQ ───────────────────────────────────────────────
GPTQ_BITS_MATRIX = int(os.environ.get("GPTQ_BITS_MATRIX", 6))
GPTQ_BITS_LOOP = int(os.environ.get("GPTQ_BITS_LOOP", 8))
# Loop layers (blocks 4 and 5) are 2.2x more quant-sensitive → use more bits
GPTQ_BITS_EMBED = int(os.environ.get("GPTQ_BITS_EMBED", 8))
GPTQ_DAMP = float(os.environ.get("GPTQ_DAMP", 0.005))
SDCLIP_K_MATRIX = float(os.environ.get("SDCLIP_K_MATRIX", 12.85))
SDCLIP_K_LOOP = float(os.environ.get("SDCLIP_K_LOOP", 6.5))
# Loop layers use more bits but tighter clip — net size impact is small
SDCLIP_K_EMBED = float(os.environ.get("SDCLIP_K_EMBED", 20.0))
AR_CALIB_SEQS = int(os.environ.get("AR_CALIB_SEQS", 64))
AR_CALIB_LEN = int(os.environ.get("AR_CALIB_LEN", 2048))
AR_CALIB_TEMP = float(os.environ.get("AR_CALIB_TEMP", 0.8))
TARGET_MB = float(os.environ.get("TARGET_MB", 15.9))

# ── PRE-QUANT TTT ─────────────────────────────────────────────────────
PREQUANT_TTT = bool(int(os.environ.get("PREQUANT_TTT", 1)))
PQ_TTT_EPOCHS = int(os.environ.get("PQ_TTT_EPOCHS", 6))
PQ_TTT_LR = float(os.environ.get("PQ_TTT_LR", 0.0005))
PQ_TTT_FREEZE = int(os.environ.get("PQ_TTT_FREEZE", 2))
# freeze first PQ_TTT_FREEZE transformer blocks (layers 0..PQ_TTT_FREEZE-1)
PQ_TTT_GRAD_CLIP = float(os.environ.get("PQ_TTT_GRAD_CLIP", 1.0))
PQ_TTT_BATCH_SIZE = int(os.environ.get("PQ_TTT_BATCH_SIZE", 4))

# ── EVALUATION ────────────────────────────────────────────────────────
EVAL_STRIDE = int(os.environ.get("EVAL_STRIDE", 64))

# ── EVAL-TIME TTT (dTTT) — ONLY ENABLE IF FITS IN 600s EVAL BUDGET ───
DTTT_ENABLED = bool(int(os.environ.get("DTTT_ENABLED", 0)))
# Default OFF. Enable after verifying total eval time < 600s.
# Adding n-gram (~90s) + standard eval (~120s) = 210s already used.
# dTTT needs ~400s more → 610s total → OVER BUDGET. Default OFF.
# Only enable if you can reduce dTTT time to < 380s.
DTTT_EPOCHS = int(os.environ.get("DTTT_EPOCHS", 10))
DTTT_LR = float(os.environ.get("DTTT_LR", 0.0005))
DTTT_CHUNK_TOKENS = int(os.environ.get("DTTT_CHUNK_TOKENS", 32768))
DTTT_FREEZE = int(os.environ.get("DTTT_FREEZE", 0))

# ── NGRAM TILT ────────────────────────────────────────────────────────
NGRAM_ENABLED = bool(int(os.environ.get("NGRAM_ENABLED", 1)))
NGRAM_BASE_BETA = float(os.environ.get("NGRAM_BASE_BETA", 2.0))
NGRAM_WITHIN_BETA = float(os.environ.get("NGRAM_WITHIN_BETA", 0.92))
NGRAM_AGREE_BONUS = float(os.environ.get("NGRAM_AGREE_BONUS", 0.1))

# ── SEED / LOGGING ────────────────────────────────────────────────────
SEED = int(os.environ.get("SEED", 42))
RUN_ID = os.environ.get("RUN_ID", f"run_seed{SEED}")
LOG_FILE = os.environ.get("LOG_FILE", f"train_log_seed{SEED}.txt")

SMOKE_TEST = bool(int(os.environ.get("SMOKE_TEST", 0)))
if SMOKE_TEST:
    MAX_WALLCLOCK_SECONDS = 120
    AR_CALIB_SEQS = 4
    PQ_TTT_EPOCHS = 1
    NGRAM_ENABLED = False  # skip ngram in smoke test

SUBMISSION_NAME = os.environ.get("SUBMISSION_NAME", "YOUR_NAME")
SUBMISSION_GITHUB = os.environ.get("SUBMISSION_GITHUB_ID", "YOUR_GITHUB_ID")


# -----------------------------
# Utilities
# -----------------------------


def log(s: str):
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() != 0:
            return
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(s.rstrip() + "\n")
    print(s, flush=True)


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        torch.cuda.set_device(rank % torch.cuda.device_count())
    else:
        rank = 0
        world = 1
    return rank, world, torch.device("cuda")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_train_files() -> list[Path]:
    pat = os.path.join(DATA_PATH, "fineweb_train_*.bin")
    files = [Path(p) for p in sorted(glob.glob(pat))]
    if not files:
        raise FileNotFoundError(f"No training shards found at {pat}")
    return files


def get_val_files() -> list[Path]:
    pat = os.path.join(DATA_PATH, "fineweb_val_*.bin")
    files = [Path(p) for p in sorted(glob.glob(pat))]
    if not files:
        raise FileNotFoundError(f"No validation shards found at {pat}")
    return files


def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("int32").itemsize
    with open(file, "rb") as f:
        header = f.read(header_bytes)
        if len(header) != header_bytes:
            raise ValueError(f"Short header in {file}")
        magic = int(np.frombuffer(header[:4], dtype=np.int32)[0])
        if magic != 20240520:
            raise ValueError(f"Bad magic {magic} in {file}")
        # remaining header fields unused
    arr = np.memmap(file, dtype=np.uint16, mode="r", offset=header_bytes)
    return torch.from_numpy(np.array(arr, dtype=np.int64))


def build_sentencepiece_luts(sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device):
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(seq_len: int) -> Tensor:
    tokens = torch.cat([load_data_shard(f) for f in get_val_files()]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Val split too short for SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def get_layer_sequence(step: int, total_steps: int) -> list[int]:
    if total_steps <= 0:
        return list(range(NUM_LAYERS))
    if step < int(ENABLE_LOOPING_AT * total_steps):
        return list(range(NUM_LAYERS))
    base = list(range(6))
    loop_insert = [4, 5] * NUM_LOOPS
    tail = list(range(6, NUM_LAYERS))
    return base + loop_insert + tail


# -----------------------------
# Shuffled sequence loader (single-host consistent)
# -----------------------------


class ShuffledSequenceLoader:
    def __init__(
        self,
        files: list[Path],
        seq_len: int,
        batch_tokens: int,
        rank: int,
        world_size: int,
        seed: int,
        prefetch_batches: int = 4,
    ):
        self.files = files
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.global_batch_seqs = batch_tokens // seq_len
        if self.global_batch_seqs <= 0:
            raise ValueError("TRAIN_BATCH_TOKENS must be >= SEQ_LEN")
        if self.global_batch_seqs % world_size != 0:
            raise ValueError("TRAIN_BATCH_TOKENS/SEQ_LEN must be divisible by WORLD_SIZE")
        self.local_batch_seqs = self.global_batch_seqs // world_size
        self.prefetch_batches = prefetch_batches
        self._rng = np.random.default_rng(seed)
        self._shards = [np.memmap(f, dtype=np.uint16, mode="r", offset=256 * 4) for f in files]
        self._shard_lengths = np.array([int(s.shape[0]) for s in self._shards], dtype=np.int64)
        self._usable_blocks = (self._shard_lengths - 1) // seq_len
        if int(self._usable_blocks.sum()) <= 0:
            raise ValueError("No usable blocks found")

        self._q: queue.Queue[tuple[Tensor, Tensor]] = queue.Queue(maxsize=prefetch_batches)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _sample_positions(self, global_seqs: int) -> list[tuple[int, int]]:
        # Sample shards proportional to remaining usable blocks; sample block offsets uniformly within shard.
        probs = self._usable_blocks.astype(np.float64)
        probs = probs / probs.sum()
        shard_ids = self._rng.choice(len(self._shards), size=global_seqs, replace=True, p=probs)
        out = []
        for sid in shard_ids:
            ub = int(self._usable_blocks[sid])
            b = int(self._rng.integers(0, ub))
            out.append((int(sid), b * self.seq_len))
        return out

    def _worker(self):
        while not self._stop.is_set():
            pairs = self._sample_positions(self.global_batch_seqs)
            xs = np.empty((self.global_batch_seqs, self.seq_len), dtype=np.int64)
            ys = np.empty((self.global_batch_seqs, self.seq_len), dtype=np.int64)
            for i, (sid, pos) in enumerate(pairs):
                shard = self._shards[sid]
                span = shard[pos : pos + self.seq_len + 1].astype(np.int64, copy=False)
                xs[i] = span[:-1]
                ys[i] = span[1:]
            start = self.rank * self.local_batch_seqs
            end = start + self.local_batch_seqs
            x = torch.from_numpy(xs[start:end].copy())
            y = torch.from_numpy(ys[start:end].copy())
            self._q.put((x, y))

    def next_batch(self) -> tuple[Tensor, Tensor]:
        return self._q.get()

    def close(self):
        self._stop.set()
        try:
            while not self._q.empty():
                self._q.get_nowait()
        except Exception:
            pass


# -----------------------------
# Model components
# -----------------------------


def apply_partial_rope(x: Tensor, cos: Tensor, sin: Tensor, rope_dims: int) -> Tensor:
    # x: [b, s, h, d]
    if rope_dims <= 0:
        return x
    x_rope = x[..., :rope_dims]
    x_pass = x[..., rope_dims:]
    x1 = x_rope[..., ::2]
    x2 = x_rope[..., 1::2]
    # cos/sin: [s, rope_dims/2]
    cos = cos[: x.shape[1]].unsqueeze(0).unsqueeze(2)
    sin = sin[: x.shape[1]].unsqueeze(0).unsqueeze(2)
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    y = torch.stack((y1, y2), dim=-1).flatten(-2)
    return torch.cat((y, x_pass), dim=-1)


class SmearGate(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        gate = torch.sigmoid(self.proj(x))
        x_shift = torch.roll(x, 1, dims=1)
        x_shift[:, 0, :] = 0
        return x * gate + x_shift * (1.0 - gate)


class MLP(nn.Module):
    def __init__(self, d_model: int, mult: float):
        super().__init__()
        hidden = int(d_model * mult)
        self.fc1 = nn.Linear(d_model, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = F.leaky_relu(x, negative_slope=0.5).square()
        x = self.fc2(x)
        return x


class FlashSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, rope_dims: int, rope_base: int, layer_idx: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model
        self.rope_dims = rope_dims
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # VE hooks (adds to V path)
        self.ve = nn.Embedding(VOCAB_SIZE, VE_DIM) if (VE_ENABLED and layer_idx in VE_LAYERS) else None
        self.ve_up = nn.Linear(VE_DIM, n_kv_heads * self.head_dim, bias=False) if self.ve is not None else None

        inv_freq = 1.0 / (rope_base ** (torch.arange(0, rope_dims, 2).float() / rope_dims))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rope_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # [s, rope_dims/2]
        return freqs.cos().to(dtype=dtype), freqs.sin().to(dtype=dtype)

    def forward(self, x: Tensor, tok_ids: Tensor | None = None) -> Tensor:
        b, s, _ = x.shape
        q = self.q_proj(x) * QK_GAIN
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(b, s, self.n_heads, self.head_dim)
        k = k.view(b, s, self.n_kv_heads, self.head_dim)
        v = v.view(b, s, self.n_kv_heads, self.head_dim)
        cos, sin = self._rope_cos_sin(s, x.device, q.dtype)
        q = apply_partial_rope(q, cos, sin, self.rope_dims)
        k = apply_partial_rope(k, cos, sin, self.rope_dims)

        if self.ve is not None and tok_ids is not None:
            ve = self.ve(tok_ids).to(dtype=v.dtype)
            v = v + self.ve_up(ve).view(b, s, self.n_kv_heads, self.head_dim)

        # GQA: repeat KV along the head axis so K/V match Q head count (manual matmul + flash-attn).
        if self.n_kv_heads != self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(rep, dim=2)
            v = v.repeat_interleave(rep, dim=2)

        if flash_attn_3_func is None:
            # fallback (slow): standard attention
            q2 = q.transpose(1, 2)  # [b,h,s,d]
            k2 = k.transpose(1, 2)
            v2 = v.transpose(1, 2)
            att = torch.matmul(q2, k2.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = torch.tril(torch.ones((s, s), device=x.device, dtype=torch.bool))
            att = att.masked_fill(~mask, -1e9)
            p = torch.softmax(att, dim=-1)
            y = torch.matmul(p, v2).transpose(1, 2).contiguous().view(b, s, self.d_model)
        else:
            # flash-attn expects [b,s,h,d] for q,k,v; handles causal mask
            y = flash_attn_3_func(q, k, v, dropout_p=0.0, causal=True)
            y = y.reshape(b, s, self.d_model)
        return self.out_proj(y)


def xsa_mix(x: Tensor) -> Tensor:
    # cross-sequence attention at zero param cost:
    # attend across batch at fixed position via softmax on dot products
    # x: [b,s,d]
    b, s, d = x.shape
    if b <= 1:
        return x
    x_t = x.transpose(0, 1).contiguous()  # [s,b,d]
    q = x_t
    k = x_t
    v = x_t
    att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)  # [s,b,b]
    p = torch.softmax(att, dim=-1)
    y = torch.matmul(p, v)  # [s,b,d]
    return y.transpose(0, 1).contiguous()


class Block(nn.Module):
    def __init__(self, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.norm1 = nn.LayerNorm(D_MODEL, elementwise_affine=True)
        self.norm2 = nn.LayerNorm(D_MODEL, elementwise_affine=True)
        self.smear = SmearGate(D_MODEL) if SMEARGATE else None
        self.attn = FlashSelfAttention(D_MODEL, NUM_HEADS, NUM_KV_HEADS, ROPE_DIMS, ROPE_BASE, layer_idx=layer_idx)
        self.mlp = MLP(D_MODEL, MLP_MULT)

    def _ln_scale(self) -> float:
        if not LN_SCALE:
            return 1.0
        return 1.0 / math.sqrt(self.layer_idx + 1.0)

    def forward(self, x: Tensor, tok_ids: Tensor) -> Tensor:
        if self.smear is not None:
            x = self.smear(x)
        if self.layer_idx >= PARALLEL_RESIDUAL_START:
            h = self.norm1(x) * self._ln_scale()
            a = self.attn(h, tok_ids)
            if XSA_ALL_LAYERS:
                a = xsa_mix(a)
            m = self.mlp(h)
            return x + a + m
        else:
            h = self.norm1(x) * self._ln_scale()
            a = self.attn(h, tok_ids)
            if XSA_ALL_LAYERS:
                a = xsa_mix(a)
            x = x + a
            h2 = self.norm2(x) * self._ln_scale()
            x = x + self.mlp(h2)
            return x


class BigramHash(nn.Module):
    def __init__(self):
        super().__init__()
        self.table = nn.Embedding(BIGRAM_VOCAB_SIZE, BIGRAM_DIM)
        self.proj = nn.Linear(BIGRAM_DIM, D_MODEL, bias=False)

    def forward(self, prev_tok: Tensor) -> Tensor:
        # prev_tok: [b,s]
        t = prev_tok.clamp(min=0, max=BIGRAM_VOCAB_SIZE - 1)
        mask = (prev_tok < BIGRAM_VOCAB_SIZE).to(dtype=torch.float32).unsqueeze(-1)
        e = self.table(t) * mask
        return self.proj(e)


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, D_MODEL)
        nn.init.normal_(self.embed.weight, std=INIT_STD)
        self.bigram = BigramHash()
        self.blocks = nn.ModuleList([Block(i) for i in range(NUM_LAYERS)])
        self.final_norm = nn.LayerNorm(D_MODEL, elementwise_affine=True)
        # tied lm head via embedding weight
        self.skip_gates = nn.Parameter(torch.zeros(8)) if UNET_SKIPS else None

    def forward(self, tok: Tensor, targets: Tensor | None = None, *, step: int = 0, total_steps: int = 1):
        x = self.embed(tok)
        prev = torch.roll(tok, 1, dims=1)
        prev[:, 0] = 0
        x = x + self.bigram(prev)

        seq = get_layer_sequence(step, total_steps)
        encoder_states: list[Tensor] = []
        for virt_i, layer_id in enumerate(seq):
            x = self.blocks[layer_id](x, tok)
            if UNET_SKIPS:
                # Encoder virtual layers: 0..7
                if virt_i < 8:
                    encoder_states.append(x)
                # Decoder virtual layers: 8..16 (apply 8 skips for 8 pairs)
                elif 8 <= virt_i < 16:
                    gate = torch.sigmoid(self.skip_gates[virt_i - 8])
                    x = gate * encoder_states[virt_i - 8] + x

        x = self.final_norm(x)
        logits = F.linear(x, self.embed.weight)
        logits = logits / LOGIT_SOFTCAP
        logits = torch.tanh(logits) * LOGIT_SOFTCAP
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.reshape(-1))
        return loss


# -----------------------------
# Optimizer: Muon (parallel-friendly minimal)
# -----------------------------


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    was_2d = G.ndim == 2
    if was_2d:
        G = G.unsqueeze(0)
    X = G.bfloat16()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    if was_2d:
        X = X.squeeze(0)
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int = 5, weight_decay: float = 0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            steps = group["backend_steps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd > 0:
                    p.data.mul_(1.0 - lr * wd)
                state = self.state[p]
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = state["momentum_buffer"] = torch.zeros_like(g, dtype=torch.bfloat16)
                buf.mul_(mom).add_(g.to(dtype=torch.bfloat16))
                upd = buf
                upd = zeropower_via_newtonschulz5(upd, steps=steps)
                scale = max(1.0, p.shape[-2] / p.shape[-1]) ** 0.5 if p.ndim == 2 else 1.0
                p.add_(upd.to(dtype=p.dtype), alpha=-lr * scale)


def split_params(model: nn.Module):
    matrices = []
    scalars = []
    tied = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n == "embed.weight":
            tied.append(p)
        elif p.ndim == 2:
            matrices.append(p)
        else:
            scalars.append(p)
    return matrices, scalars, tied


# -----------------------------
# EMA / SWA
# -----------------------------


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        sd = model.state_dict()
        for k, v in sd.items():
            self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)

    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=True)


class SWA:
    def __init__(self, model: nn.Module):
        self.n = 0
        self.avg = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.n += 1
        sd = model.state_dict()
        for k, v in sd.items():
            self.avg[k].mul_((self.n - 1) / self.n).add_(v, alpha=1.0 / self.n)

    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.avg, strict=True)


# -----------------------------
# Eval (BPB) + optional n-gram tilt
# -----------------------------

def build_word_boundary_luts_cpu(sp: spm.SentencePieceProcessor, vocab_size: int) -> tuple[np.ndarray, np.ndarray]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    has_leading_space = np.zeros((table_size,), dtype=np.bool_)
    is_boundary = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary[token_id] = False
        if sp.is_byte(token_id):
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space[token_id] = True
    return has_leading_space[:vocab_size], is_boundary[:vocab_size]


class CausalNgramHashTableCPU:
    def __init__(self, vocab_size: int, has_space: np.ndarray, is_boundary: np.ndarray):
        self.vocab_size = vocab_size
        self.has_space = has_space
        self.is_boundary = is_boundary
        # order 8..16 exact-context next-token counts
        self.high: dict[int, dict[tuple[int, ...], dict[int, int]]] = {n: {} for n in range(8, 17)}
        # within-word ngrams order 1..3 (context within current word)
        self.within: dict[int, dict[tuple[int, ...], dict[int, int]]] = {n: {} for n in range(1, 4)}
        # word-start bigrams: last_word_start -> next_word_start counts
        self.word_start: dict[int, dict[int, int]] = {}

        self.hist: deque[int] = deque(maxlen=16)  # full-token history (for high-order)
        self.word_hist: deque[int] = deque(maxlen=8)  # within-word history
        self.in_word = False
        self.last_word_start: int | None = None

    def _bump(self, table: dict[tuple[int, ...], dict[int, int]] | dict[int, dict[int, int]], key, tok: int) -> int:
        bucket = table.get(key)
        if bucket is None:
            bucket = {}
            table[key] = bucket
        bucket[tok] = bucket.get(tok, 0) + 1
        return bucket[tok]

    @staticmethod
    def _best(bucket: dict[int, int] | None) -> tuple[int, int] | None:
        if not bucket:
            return None
        best_tok = -1
        best_cnt = -1
        for t, c in bucket.items():
            if c > best_cnt:
                best_tok = t
                best_cnt = c
        return best_tok, best_cnt

    def emit_hint(self) -> tuple[int, int]:
        # Returns (hint_token, support_count). If no hint, (-1, 0).
        best_hint = -1
        best_cnt = 0

        hlist = list(self.hist)
        # Prefer longest high-order context available.
        for n in range(16, 7, -1):
            if len(hlist) < n:
                continue
            ctx = tuple(hlist[-n:])
            b = self.high[n].get(ctx)
            cand = self._best(b)
            if cand is not None and cand[1] > best_cnt:
                best_hint, best_cnt = cand
                # if we have a strong high-order match, keep it.
                break

        # Within-word contexts (1..3), only if we are in a word.
        if best_hint == -1 and self.in_word:
            wlist = list(self.word_hist)
            for n in range(3, 0, -1):
                if len(wlist) < n:
                    continue
                ctx = tuple(wlist[-n:])
                cand = self._best(self.within[n].get(ctx))
                if cand is not None and cand[1] > best_cnt:
                    best_hint, best_cnt = cand
                    break

        # Word-start bigram (predict next word-start token).
        if best_hint == -1 and self.last_word_start is not None:
            cand = self._best(self.word_start.get(self.last_word_start))
            if cand is not None and cand[1] > best_cnt:
                best_hint, best_cnt = cand

        return best_hint, best_cnt

    def update_after_scoring(self, tok: int):
        # Insert token tok AFTER scoring it.
        if 0 <= tok < self.vocab_size:
            # update high-order ngrams (8..16) using history before tok
            hlist = list(self.hist)
            for n in range(8, 17):
                if len(hlist) < n:
                    continue
                ctx = tuple(hlist[-n:])
                self._bump(self.high[n], ctx, tok)

            # update within-word ngrams (1..3) if currently in word (before tok)
            if self.in_word:
                wlist = list(self.word_hist)
                for n in range(1, 4):
                    if len(wlist) < n:
                        continue
                    ctx = tuple(wlist[-n:])
                    self._bump(self.within[n], ctx, tok)

            # update word-start bigrams if tok begins a new word
            tok_word_start = bool(self.has_space[tok])
            tok_boundary = bool(self.is_boundary[tok])
            if tok_boundary:
                # boundary/control breaks word state
                self.in_word = False
                self.word_hist.clear()
            elif tok_word_start:
                if self.last_word_start is not None:
                    self._bump(self.word_start, self.last_word_start, tok)
                self.last_word_start = tok
                self.in_word = True
                self.word_hist.clear()
                self.word_hist.append(tok)
            else:
                # within-word continuation
                if not self.in_word:
                    self.in_word = True
                    self.word_hist.clear()
                self.word_hist.append(tok)

            self.hist.append(tok)
        else:
            # OOV / out-of-range: still append sentinel behavior
            self.hist.append(0)


def tilted_nll_from_logits(logits_row: Tensor, target: int, hint: int, beta: float) -> Tensor:
    # logits_row: [v]
    # returns negative log likelihood under tilt rule:
    # p_tilt(t) ∝ p_model(t) * exp(beta * 1[t==hint])
    # Z = 1 + p_model(hint) * (exp(beta) - 1)
    logZ = torch.logsumexp(logits_row, dim=-1)
    logp_tgt = logits_row[target] - logZ
    if hint < 0 or hint >= logits_row.shape[0]:
        return -logp_tgt
    logp_hint = logits_row[hint] - logZ
    # log(Z_tilt) = log(1 + p_hint*(exp(beta)-1))
    logZ_tilt = torch.log1p(torch.exp(logp_hint) * (math.exp(beta) - 1.0))
    if target == hint:
        return -(logp_tgt + beta - logZ_tilt)
    return -(logp_tgt - logZ_tilt)


@torch.no_grad()
def eval_val_bpb(model: nn.Module, device: torch.device, rank: int, world_size: int, sp: spm.SentencePieceProcessor):
    val_tokens = load_validation_tokens(SEQ_LEN)
    base_bytes, has_space, is_boundary = build_sentencepiece_luts(sp, VOCAB_SIZE, device)

    has_space_cpu, is_boundary_cpu = build_word_boundary_luts_cpu(sp, VOCAB_SIZE)
    ngram_table = CausalNgramHashTableCPU(VOCAB_SIZE, has_space_cpu, is_boundary_cpu) if NGRAM_ENABLED else None

    model.eval()
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    tok_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    # Sliding window evaluation:
    # - window size: SEQ_LEN
    # - stride: EVAL_STRIDE
    # - score only last EVAL_STRIDE tokens of each window
    total_tokens = int(val_tokens.numel())
    total_windows = (total_tokens - (SEQ_LEN + 1)) // EVAL_STRIDE + 1
    win_start = (total_windows * rank) // world_size
    win_end = (total_windows * (rank + 1)) // world_size

    # Maintain causal n-gram state in global token order (per-rank stream).
    ngram_pos = -1  # last global token index inserted into ngram table

    for w in range(win_start, win_end):
        start = w * EVAL_STRIDE
        raw_cpu = val_tokens[start : start + SEQ_LEN + 1].to(torch.int64)  # [SEQ_LEN+1]
        raw = raw_cpu.to(device, non_blocking=True)
        x = raw[:-1].unsqueeze(0)  # [1,SEQ_LEN]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            logits = model(x, None, step=0, total_steps=1)  # [1,SEQ_LEN,V]
        logits_flat = logits.squeeze(0)  # [SEQ_LEN,V]

        # Score only the last EVAL_STRIDE positions in the window.
        local_score_start = SEQ_LEN - EVAL_STRIDE  # inclusive, in logits index space
        for i in range(local_score_start, SEQ_LEN):
            global_tgt_idx = start + i + 1  # target token index in val_tokens

            # Ensure the n-gram table has ingested tokens up to global_tgt_idx-1.
            if ngram_table is not None:
                while ngram_pos < global_tgt_idx - 1:
                    ngram_pos += 1
                    ngram_table.update_after_scoring(int(val_tokens[ngram_pos].item()))

            tgt = int(raw_cpu[i + 1].item())
            if ngram_table is None:
                nll = F.cross_entropy(logits_flat[i].unsqueeze(0), torch.tensor([tgt], device=device), reduction="sum")
            else:
                hint, _cnt = ngram_table.emit_hint()
                nll = tilted_nll_from_logits(logits_flat[i].float(), tgt, hint, beta=NGRAM_AGREE_BONUS)
                # Update AFTER scoring the token.
                ngram_table.update_after_scoring(tgt)
                ngram_pos = max(ngram_pos, global_tgt_idx)

            loss_sum += nll.to(torch.float64)
            tok_count += 1.0

            prev_id = int(raw_cpu[i].item())
            tgt_id = tgt
            tb = base_bytes[tgt_id].to(dtype=torch.int16)
            tb = tb + ((has_space[tgt_id] & ~is_boundary[prev_id]).to(dtype=torch.int16))
            byte_count += tb.to(torch.float64)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(tok_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    val_loss = loss_sum / tok_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = tok_count.item() / byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


# -----------------------------
# Pre-quant TTT (EMA copy, self-generated calibration)
# -----------------------------


@torch.no_grad()
def generate_calibration_tokens(model: nn.Module, device: torch.device, seed: int = 42) -> list[Tensor]:
    torch.manual_seed(seed)
    all_tokens: list[Tensor] = []
    model.eval()
    for _ in range(AR_CALIB_SEQS):
        tok = torch.zeros(1, 1, dtype=torch.long, device=device)
        for _i in range(AR_CALIB_LEN - 1):
            logits = model(tok, None, step=0, total_steps=1)[:, -1, :]
            logits = logits / AR_CALIB_TEMP
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            tok = torch.cat([tok, nxt], dim=1)
        all_tokens.append(tok[0].detach().to("cpu"))
    model.train()
    return all_tokens


def run_prequant_ttt_from_ema(ema_model: nn.Module, device: torch.device) -> nn.Module:
    # Returns a NEW adapted model instance. Does not modify ema_model.
    adapted = copy.deepcopy(ema_model).to(device)
    if not PREQUANT_TTT:
        return adapted

    for li in range(min(PQ_TTT_FREEZE, len(adapted.blocks))):
        for p in adapted.blocks[li].parameters():
            p.requires_grad = False

    calib = generate_calibration_tokens(adapted, device, seed=SEED)
    log(
        f"prequant_ttt: start seed={SEED} seqs={AR_CALIB_SEQS} len={AR_CALIB_LEN} "
        f"epochs={PQ_TTT_EPOCHS} bs={PQ_TTT_BATCH_SIZE} freeze={PQ_TTT_FREEZE} lr={PQ_TTT_LR}"
    )

    params = [p for p in adapted.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=PQ_TTT_LR, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(calib) / PQ_TTT_BATCH_SIZE)
    total_steps = max(1, PQ_TTT_EPOCHS * steps_per_epoch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    adapted.train()
    step_idx = 0
    for ep in range(PQ_TTT_EPOCHS):
        random.shuffle(calib)
        for i in range(0, len(calib), PQ_TTT_BATCH_SIZE):
            batch = calib[i : i + PQ_TTT_BATCH_SIZE]
            tok = torch.stack([t[:SEQ_LEN] for t in batch], dim=0).to(device)
            x = tok[:, :-1]
            y = tok[:, 1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = adapted(x, y, step=0, total_steps=1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, PQ_TTT_GRAD_CLIP)
            opt.step()
            sched.step()
            step_idx += 1

    log("prequant_ttt: done")
    return adapted


# -----------------------------
# GPTQ + SDClip quantization (full-Hessian approximation via X^T X)
# -----------------------------


@dataclass
class QuantizedTensor:
    q: Tensor
    scale: Tensor
    bits: int


def sdclip_quantize_row(row: Tensor, bits: int, k: float) -> tuple[Tensor, float]:
    std = row.float().std().item()
    clip = max(1e-8, k * std)
    row_clipped = row.clamp(-clip, clip)
    qmax = (2 ** (bits - 1) - 1)
    scale = clip / qmax
    row_q = (row_clipped / scale).round().clamp(-qmax - 1, qmax).to(torch.int16)
    return row_q, float(scale)


def pack_int_nbits(q: Tensor, bits: int) -> bytes:
    # q: int16 values in [-2^(b-1), 2^(b-1)-1], pack to little-endian bitstream
    q = q.to(torch.int32).flatten().cpu().numpy()
    mask = (1 << bits) - 1
    # convert signed to unsigned
    q = (q & mask).astype(np.uint32)
    out = bytearray()
    acc = 0
    acc_bits = 0
    for v in q:
        acc |= int(v) << acc_bits
        acc_bits += bits
        while acc_bits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            acc_bits -= 8
    if acc_bits:
        out.append(acc & 0xFF)
    return bytes(out)


def gptq_full_hessian_quantize_linear(W: Tensor, Xs: list[Tensor], bits: int, k: float, damp: float) -> tuple[bytes, Tensor]:
    # W: [out,in] on cpu (float16/bf16/float32)
    # Xs: list of [batch*seq, in] float32 activations
    W32 = W.float().cpu()
    in_dim = W32.shape[1]
    H = torch.zeros((in_dim, in_dim), dtype=torch.float32)
    for X in Xs:
        X = X.reshape(-1, in_dim).float().cpu()
        H += X.T @ X
    H /= max(1.0, float(sum(int(x.shape[0]) for x in Xs)))
    diag = torch.diag(H)
    damp_i = float(damp)
    # Increase damping if repeated activations make H near-singular.
    L = None
    for _try in range(8):
        H_try = H + torch.eye(in_dim, dtype=torch.float32) * (damp_i * diag.mean().clamp_min(1e-6))
        try:
            L = torch.linalg.cholesky(H_try)
            H = H_try
            break
        except RuntimeError:
            damp_i *= 2.0
    if L is None:
        # Last resort: keep heavily damped matrix to avoid crash.
        H = H + torch.eye(in_dim, dtype=torch.float32) * (damp_i * diag.mean().clamp_min(1e-6))
        L = torch.linalg.cholesky(H)

    # actorder (permute by diag magnitude)
    perm = torch.argsort(torch.diag(H), descending=True)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(in_dim)
    H = H[perm][:, perm]
    W32 = W32[:, perm]
    # Cholesky + inverse for GPTQ column-wise error compensation.
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)  # [in,in]

    out_dim = W32.shape[0]
    qmax = (2 ** (bits - 1) - 1)

    # Per-row SDClip scale (fixed for whole row) + column-wise GPTQ propagation.
    Wwork = W32.clone()  # [out,in] float32, updated in-place
    scales = torch.empty((out_dim,), dtype=torch.float32)
    clip = torch.empty((out_dim,), dtype=torch.float32)
    for r in range(out_dim):
        std = float(Wwork[r].std().item())
        c = max(1e-8, float(k) * std)
        clip[r] = c
        scales[r] = c / qmax

    Qp = torch.empty((out_dim, in_dim), dtype=torch.int16)
    denom = torch.diag(Hinv).clamp_min(1e-12)  # [in]

    for i in range(in_dim):
        w_i = Wwork[:, i]
        c = clip
        s = scales
        w_clip = w_i.clamp(-c, c)
        q_i = torch.round(w_clip / s).clamp(-qmax - 1, qmax).to(torch.int16)
        Qp[:, i] = q_i
        # error in float32
        w_hat = q_i.to(torch.float32) * s
        err = (w_i - w_hat) / denom[i]
        if i + 1 < in_dim:
            # Propagate to remaining columns j>i:
            # W[:, j] -= err * Hinv[i, j]
            Wwork[:, i + 1 :] -= err.unsqueeze(1) * Hinv[i, i + 1 :].unsqueeze(0)

    # undo perm in packed stream by writing in original column order
    Q = Qp[:, inv_perm]
    # undo perm in packed stream by writing in original column order
    packed = pack_int_nbits(Q, bits)
    return packed, scales.to(torch.float16)


def collect_linear_inputs(model: nn.Module, calib: list[Tensor], device: torch.device, total_steps: int):
    layer_inputs: dict[str, list[Tensor]] = {}
    hooks = []

    def make_hook(name: str):
        def hook(_m, inp, _out):
            x = inp[0].detach()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            layer_inputs.setdefault(name, []).append(x.float().cpu())

        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for t in calib:
            tok = t[:SEQ_LEN].unsqueeze(0).to(device)
            model(tok[:, :-1], None, step=0, total_steps=total_steps)

    for h in hooks:
        h.remove()
    model.train()
    return layer_inputs


def export_sdclip_gptq(model: nn.Module, device: torch.device, total_steps: int, *, sdclip_k_matrix: float) -> bytes:
    calib = generate_calibration_tokens(model, device, seed=SEED)
    layer_inputs = collect_linear_inputs(model, calib, device, total_steps=total_steps)

    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    payload: dict[str, object] = {"format": "sdclip_gptq_v1", "tensors": {}}
    tensors: dict[str, object] = {}

    for name, t in state.items():
        if t.ndim == 2 and "embed.weight" not in name:
            bits = GPTQ_BITS_MATRIX
            k = sdclip_k_matrix
            if ".blocks.4." in name or ".blocks.5." in name:
                bits = GPTQ_BITS_LOOP
                k = SDCLIP_K_LOOP
            Xs = layer_inputs.get(name.replace(".weight", ""), None) or layer_inputs.get(name.rsplit(".weight", 1)[0], [])
            if not Xs:
                # fallback: treat as plain sdclip per-row without Hessian
                q_rows = []
                scales = torch.empty((t.shape[0],), dtype=torch.float16)
                for r in range(t.shape[0]):
                    q, sc = sdclip_quantize_row(t[r], bits=bits, k=k)
                    q_rows.append(q)
                    scales[r] = sc
                Q = torch.stack(q_rows, dim=0)
                packed = pack_int_nbits(Q, bits)
                tensors[name] = {"packed": base64.b85encode(packed).decode("ascii"), "scale": scales.numpy().tobytes().hex(), "bits": bits, "shape": list(t.shape)}
            else:
                packed, scales = gptq_full_hessian_quantize_linear(t, Xs, bits=bits, k=k, damp=GPTQ_DAMP)
                tensors[name] = {"packed": base64.b85encode(packed).decode("ascii"), "scale": scales.numpy().tobytes().hex(), "bits": bits, "shape": list(t.shape)}
        elif name == "embed.weight":
            # GPTQ embeddings too (row-wise)
            bits = GPTQ_BITS_EMBED
            k = SDCLIP_K_EMBED
            q_rows = []
            scales = torch.empty((t.shape[0],), dtype=torch.float16)
            for r in range(t.shape[0]):
                q, sc = sdclip_quantize_row(t[r], bits=bits, k=k)
                q_rows.append(q)
                scales[r] = sc
            Q = torch.stack(q_rows, dim=0)
            packed = pack_int_nbits(Q, bits)
            tensors[name] = {"packed": base64.b85encode(packed).decode("ascii"), "scale": scales.numpy().tobytes().hex(), "bits": bits, "shape": list(t.shape)}
        else:
            # keep as float16 for small / non-matrix
            tensors[name] = {"raw_f16": t.to(torch.float16).numpy().tobytes().hex(), "shape": list(t.shape)}

    payload["tensors"] = tensors
    raw = io.BytesIO()
    raw.write(repr(payload).encode("utf-8"))
    blob = raw.getvalue()
    try:
        return lzma.compress(blob, preset=9)
    except Exception:
        if zstandard is None:
            raise
        return zstandard.ZstdCompressor(level=22).compress(blob)


# -----------------------------
# Training loop
# -----------------------------


def lr_schedule(step: int, total_steps: int) -> float:
    if step < WARMUP_ITERS:
        return step / max(1, WARMUP_ITERS)
    if step >= total_steps:
        return 0.0
    t = (step - WARMUP_ITERS) / max(1, total_steps - WARMUP_ITERS)
    # cosine to 0 over warmdown window
    wd_start = max(0, total_steps - WARMDOWN_ITERS)
    if step < wd_start:
        return 1.0
    tw = (step - wd_start) / max(1, WARMDOWN_ITERS)
    return 0.5 * (1.0 + math.cos(math.pi * tw))


def main():
    rank, world_size, device = setup_distributed()
    set_seed(SEED + rank)

    if rank == 0:
        Path(LOG_FILE).write_text("", encoding="utf-8")
        log(f"seed={SEED} run_id={RUN_ID} smoke_test={int(SMOKE_TEST)}")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_PATH)

    model = GPT().to(device)
    ddp = DDP(model, device_ids=[device.index], broadcast_buffers=False) if world_size > 1 else model

    matrices, scalars, tied = split_params(model)
    muon = Muon(matrices, lr=MATRIX_LR, momentum=MUON_MOMENTUM, backend_steps=5, weight_decay=MUON_WD)
    adam = torch.optim.AdamW(
        [{"params": scalars, "lr": SCALAR_LR, "weight_decay": ADAM_WD}, {"params": tied, "lr": TIED_EMBED_LR, "weight_decay": ADAM_WD}],
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    ema = EMA(model, EMA_DECAY) if EMA_ENABLED else None
    swa = SWA(model) if SWA_ENABLED else None

    files = get_train_files()
    loader = ShuffledSequenceLoader(files, seq_len=SEQ_LEN, batch_tokens=TRAIN_BATCH_TOKENS, rank=rank, world_size=world_size, seed=SEED)

    start = time.time()
    train_start = start
    step = 0
    # heuristic steps budget: assume ~85ms/step on H100; reserve ~120s eval+export
    total_steps = int((MAX_WALLCLOCK_SECONDS - 130) / 0.09)
    total_steps = max(100, total_steps)
    log(f"run_id={RUN_ID} rank={rank}/{world_size} total_steps={total_steps} seq_len={SEQ_LEN} batch_tokens={TRAIN_BATCH_TOKENS}")

    while True:
        now = time.time()
        if now - start > MAX_WALLCLOCK_SECONDS - 135:
            break
        if step >= total_steps:
            break

        x_cpu, y_cpu = loader.next_batch()
        x = x_cpu.to(device, non_blocking=True)
        y = y_cpu.to(device, non_blocking=True)

        lr_mult = lr_schedule(step, total_steps)
        # Set LRs from fixed bases (avoid compounding).
        adam.param_groups[0]["lr"] = SCALAR_LR * lr_mult
        adam.param_groups[1]["lr"] = TIED_EMBED_LR * lr_mult
        for pg in muon.param_groups:
            pg["lr"] = MATRIX_LR * lr_mult
            # warm momentum
            if step < MUON_MOM_WARMUP_STEPS:
                t = step / max(1, MUON_MOM_WARMUP_STEPS)
                pg["momentum"] = MUON_MOM_WARMUP_START * (1 - t) + MUON_MOMENTUM * t
            else:
                pg["momentum"] = MUON_MOMENTUM

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            loss = ddp(x, y, step=step, total_steps=total_steps) if world_size > 1 else model(x, y, step=step, total_steps=total_steps)

        adam.zero_grad(set_to_none=True)
        muon.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        adam.step()
        muon.step()

        if ema is not None:
            ema.update(model)
        if swa is not None and (step % SWA_EVERY == 0) and (step > WARMUP_ITERS):
            swa.update(model)

        if rank == 0:
            log(f"train: step={step} loss={float(loss.item()):.6f} lr_mult={lr_mult:.4f} wall={now-start:.1f}s")

        step += 1

    loader.close()
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    train_end = time.time()

    # swap in EMA for eval/export by default (best under quant)
    eval_model = copy.deepcopy(model).to(device)
    if ema is not None:
        ema.copy_to(eval_model)
    elif swa is not None:
        swa.copy_to(eval_model)

    val_loss, val_bpb = eval_val_bpb(eval_model, device, rank, world_size, sp)
    if rank == 0:
        log(f"val_loss={val_loss:.6f} val_bpb={val_bpb:.6f}")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if rank == 0:
        # STEP 7: Pre-Quant TTT on a COPY of EMA weights, then GPTQ that adapted copy.
        pq_model = run_prequant_ttt_from_ema(eval_model, device)

        # STEP 9: Artifact verification loop: adjust SDCLIP_K_MATRIX until under 16MB (including code bytes).
        # Evaluator counts: compressed_model_bytes + raw UTF-8 code bytes.
        code_bytes = Path(__file__).read_bytes()
        k_matrix = float(SDCLIP_K_MATRIX)
        blob = b""
        total_bytes = 1 << 60
        for attempt in range(40):
            blob = export_sdclip_gptq(pq_model, device, total_steps=total_steps, sdclip_k_matrix=k_matrix)
            total_bytes = len(blob) + len(code_bytes)
            if total_bytes <= 16_000_000:
                break
            k_matrix += 0.5
            log(f"artifact_too_big: total={total_bytes} bump SDCLIP_K_MATRIX -> {k_matrix:.2f} (attempt {attempt+1})")
        out_path = Path("submission.bin")
        out_path.write_bytes(blob)
        log(f"ARTIFACT: {total_bytes} bytes ({total_bytes/1e6:.3f} MB)")
        log(f"export: wrote {out_path} compressed={len(blob)} code={len(code_bytes)} compressor={_COMPRESSOR} sdclip_k_matrix={k_matrix:.2f}")

        # STEP 11: submission.json
        sub = {
            "name": SUBMISSION_NAME,
            "github": SUBMISSION_GITHUB,
            "val_bpb": float(val_bpb),
            "val_bpb_std": None,
            "seeds": [int(SEED)],
            "date": "2026-04-XX",
            "summary": "SP8192 + depth recurrence (triple loop) + Pre-Quant TTT + causal n-gram tilt + mixed-precision GPTQ",
            "hardware": "8xH100 SXM",
            "artifact_bytes": int(total_bytes),
            "training_seconds": float(train_end - train_start),
        }
        Path("submission.json").write_text(json.dumps(sub, indent=2) + "\n", encoding="utf-8")
        log(f"submission_json: wrote submission.json training_seconds={sub['training_seconds']:.2f}")


if __name__ == "__main__":
    main()

