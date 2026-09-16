#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fusion/training-protocol experiment suite for wearable radar-IMU hand gesture recognition.

Purpose
-------
This script is designed for the revised paper direction:
  - NOT a new neural-network architecture paper.
  - A fair comparison of radar/IMU fusion or training protocols under the same
    subject-disjoint wearable head-motion setting.

Main comparison methods
-----------------------
1) radar_only
   Radar-derived polar/time-series features only. No IMU features.
2) early_fusion
   Radar features and IMU quaternion-derived features concatenated at every time step.
3) late_fusion
   Radar and IMU sequences are encoded by separate encoders, then fused in latent space.
4) attention_fusion
   Radar and IMU sequences are encoded separately, and IMU features guide radar features
   through cross-attention.
5) motion_domain_proposed
   Doppler-aware motion-XYZ representation produced from geometry-corrected radar vectors.

Backbones
---------
- tcn
- gru
- transformer

The code intentionally avoids multi-worker DataLoader to prevent "too many open files" errors
and to make repeated fold/seed experiments faster.
"""
from __future__ import annotations

import os
import sys
import glob
import json
import math
import time
import csv
import hashlib
import argparse
import traceback
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

LABELS_4 = ["01RtoL", "02LtoR", "03UtoD", "04DtoU"]
LABELS_5 = ["01RtoL", "02LtoR", "03UtoD", "04DtoU", "05None"]

HEAD_ALIASES = {
    "none": "None",
    "None": "None",
    "NONE": "None",
    "down": "Down",
    "Down": "Down",
    "DOWN": "Down",
    "left": "Left",
    "Left": "Left",
    "LEFT": "Left",
    "right": "Right",
    "Right": "Right",
    "RIGHT": "Right",
    "up": "UP",
    "Up": "UP",
    "UP": "UP",
    "random": "Random",
    "Random": "Random",
    "RANDOM": "Random",
}

CANON_HEADS = ["None", "Down", "Left", "Random", "Right", "UP"]

RADAR_BASE_COLS = ["range", "doppler", "horizontal_angle", "vertical_angle"]
QUAT_COLS = ["q.w", "q.i", "q.j", "q.k"]


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def short_hash(text: str, n: int = 12) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:n]


def canonical_head(h: str) -> str:
    return HEAD_ALIASES.get(str(h), str(h))


def safe_float_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr[~np.isfinite(arr)] = np.nan
    return arr


def interp_nan_1d(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0:
        return y
    finite = np.isfinite(y)
    if finite.all():
        return y
    if not finite.any():
        return np.zeros_like(y, dtype=np.float64)
    x = np.arange(len(y), dtype=np.float64)
    y2 = y.copy()
    y2[~finite] = np.interp(x[~finite], x[finite], y[finite])
    return y2


def resample_matrix(mat: np.ndarray, steps: int) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError("mat must be 2D")
    if mat.shape[0] == 0:
        return np.zeros((steps, mat.shape[1]), dtype=np.float32)
    if mat.shape[0] == 1:
        return np.repeat(mat.astype(np.float32), steps, axis=0)
    old_x = np.linspace(0.0, 1.0, mat.shape[0])
    new_x = np.linspace(0.0, 1.0, steps)
    out = []
    for c in range(mat.shape[1]):
        col = interp_nan_1d(mat[:, c])
        out.append(np.interp(new_x, old_x, col))
    return np.stack(out, axis=1).astype(np.float32)


def sincos_deg(angle_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rad = np.deg2rad(angle_deg)
    return np.sin(rad), np.cos(rad)


def polar_to_xyz(rng: np.ndarray, h_deg: np.ndarray, v_deg: np.ndarray) -> np.ndarray:
    h = np.deg2rad(h_deg)
    v = np.deg2rad(v_deg)
    x = rng * np.cos(v) * np.cos(h)
    y = rng * np.cos(v) * np.sin(h)
    z = rng * np.sin(v)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def normalize_quat(q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=1, keepdims=True)
    qn = np.zeros_like(q)
    valid = n[:, 0] > eps
    qn[valid] = q[valid] / n[valid]
    qn[~valid, 0] = 1.0
    return qn.astype(np.float32)


def quat_conj(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[:, 1:] *= -1
    return out


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = [q1[:, i] for i in range(4)]
    w2, x2, y2, z2 = [q2[:, i] for i in range(4)]
    out = np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=1)
    return normalize_quat(out)


def quaternion_relative_to_first(q: np.ndarray) -> np.ndarray:
    q = normalize_quat(q)
    q0 = np.repeat(q[:1], len(q), axis=0)
    return quat_mul(quat_conj(q0), q)


def delta(mat: np.ndarray) -> np.ndarray:
    return np.diff(mat, axis=0, prepend=mat[:1]).astype(np.float32)


# -----------------------------------------------------------------------------
# Dataset feature extraction
# -----------------------------------------------------------------------------

@dataclass
class CachedDataset:
    radar: np.ndarray
    imu: np.ndarray
    motion: np.ndarray
    labels: np.ndarray
    persons: np.ndarray
    heads: np.ndarray
    files: np.ndarray
    label_names: List[str]
    radar_feature_names: List[str]
    imu_feature_names: List[str]
    motion_feature_names: List[str]


def choose_col(df: pd.DataFrame, preferred: str, fallback: str) -> np.ndarray:
    col = preferred if preferred in df.columns else fallback
    if col not in df.columns:
        return np.zeros(len(df), dtype=np.float64)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)


def read_csv_features(path: str, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)

    # Radar polar sequence. Prefer corrected columns if present.
    rng = choose_col(df, "range_corr", "range")
    dop = choose_col(df, "doppler_corr", "doppler")
    h = choose_col(df, "horizontal_angle_corr", "horizontal_angle")
    v = choose_col(df, "vertical_angle_corr", "vertical_angle")
    hs, hc = sincos_deg(h)
    vs, vc = sincos_deg(v)
    radar_base = np.stack([rng, dop, hs, hc, vs, vc], axis=1)
    radar_base = resample_matrix(radar_base, steps)
    radar_delta = delta(radar_base)
    # Keep compact but informative radar-only features.
    radar = np.concatenate([
        radar_base,
        radar_delta[:, 0:2],    # drange, ddoppler
        radar_delta[:, 2:6],    # angular sin/cos changes
    ], axis=1).astype(np.float32)

    # IMU quaternion-derived features.
    if all(c in df.columns for c in QUAT_COLS):
        q_raw = np.stack([pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64) for c in QUAT_COLS], axis=1)
    else:
        q_raw = np.zeros((len(df), 4), dtype=np.float64)
        q_raw[:, 0] = 1.0
    q = resample_matrix(q_raw, steps)
    q = normalize_quat(q)
    q_rel = quaternion_relative_to_first(q)
    dq = delta(q)
    imu = np.concatenate([q, q_rel, dq], axis=1).astype(np.float32)

    # Motion-domain proposed features. Prefer selected reference/world vector columns.
    xyz_cols = ["radar_x_world", "radar_y_world", "radar_z_world"]
    if all(c in df.columns for c in xyz_cols):
        xyz = np.stack([pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64) for c in xyz_cols], axis=1)
        xyz = resample_matrix(xyz, steps)
    else:
        polar_mat = np.stack([rng, h, v], axis=1)
        polar_mat = resample_matrix(polar_mat, steps)
        xyz = polar_to_xyz(polar_mat[:, 0], polar_mat[:, 1], polar_mat[:, 2])
    rd = np.stack([rng, dop], axis=1)
    rd = resample_matrix(rd, steps)
    r_rs = rd[:, 0]
    d_rs = rd[:, 1]
    xyz_rel = xyz - xyz[:1]
    dxyz = delta(xyz_rel)
    speed = np.linalg.norm(dxyz, axis=1, keepdims=True).astype(np.float32)
    ddop = delta(d_rs.reshape(-1, 1))
    r_rel = (r_rs - r_rs[0]).reshape(-1, 1).astype(np.float32)
    dr = delta(r_rs.reshape(-1, 1))
    motion = np.concatenate([
        xyz_rel.astype(np.float32),      # 3
        dxyz.astype(np.float32),         # 3
        speed.astype(np.float32),        # 1
        d_rs.reshape(-1, 1).astype(np.float32),  # 1 doppler
        ddop.astype(np.float32),         # 1
        r_rel.astype(np.float32),        # 1
        dr.astype(np.float32),           # 1
    ], axis=1).astype(np.float32)

    return radar, imu, motion


def build_or_load_dataset(data_root: str, result_root: str, class_mode: int, steps: int, force_rebuild: bool = False) -> CachedDataset:
    labels_allowed = LABELS_4 if class_mode == 4 else LABELS_5
    cache_dir = os.path.join(result_root, "cache")
    ensure_dir(cache_dir)
    key = short_hash(os.path.abspath(data_root) + f"|class={class_mode}|steps={steps}|v=fusion2")
    cache_path = os.path.join(cache_dir, f"cached_dataset_{key}.npz")
    meta_path = os.path.join(cache_dir, f"cached_dataset_{key}.json")

    if os.path.isfile(cache_path) and os.path.isfile(meta_path) and not force_rebuild:
        npz = np.load(cache_path, allow_pickle=True)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        print(f"[CACHE LOAD] {cache_path}")
        return CachedDataset(
            radar=npz["radar"].astype(np.float32),
            imu=npz["imu"].astype(np.float32),
            motion=npz["motion"].astype(np.float32),
            labels=npz["labels"].astype(np.int64),
            persons=npz["persons"].astype(str),
            heads=npz["heads"].astype(str),
            files=npz["files"].astype(str),
            label_names=meta["label_names"],
            radar_feature_names=meta["radar_feature_names"],
            imu_feature_names=meta["imu_feature_names"],
            motion_feature_names=meta["motion_feature_names"],
        )

    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"DATA_ROOT does not exist: {data_root}")

    radar_list, imu_list, motion_list = [], [], []
    y_list, person_list, head_list, file_list = [], [], [], []
    considered, skipped = 0, 0
    label_to_idx = {name: i for i, name in enumerate(labels_allowed)}

    for person in sorted(os.listdir(data_root)):
        person_dir = os.path.join(data_root, person)
        if not os.path.isdir(person_dir):
            continue
        for gesture in sorted(os.listdir(person_dir)):
            if gesture not in label_to_idx:
                continue
            gesture_dir = os.path.join(person_dir, gesture)
            if not os.path.isdir(gesture_dir):
                continue
            for head_raw in sorted(os.listdir(gesture_dir)):
                head_dir = os.path.join(gesture_dir, head_raw)
                if not os.path.isdir(head_dir):
                    continue
                head = canonical_head(head_raw)
                csv_files = sorted(glob.glob(os.path.join(head_dir, "**", "*.csv"), recursive=True))
                for path in csv_files:
                    considered += 1
                    try:
                        radar, imu, motion = read_csv_features(path, steps)
                        radar_list.append(radar)
                        imu_list.append(imu)
                        motion_list.append(motion)
                        y_list.append(label_to_idx[gesture])
                        person_list.append(person)
                        head_list.append(head)
                        file_list.append(path)
                    except Exception as e:
                        skipped += 1
                        print(f"[SKIP] {path}: {e}")

    if not radar_list:
        raise RuntimeError("No valid CSV files were loaded. Check data path and folder structure.")

    radar_arr = np.stack(radar_list).astype(np.float32)
    imu_arr = np.stack(imu_list).astype(np.float32)
    motion_arr = np.stack(motion_list).astype(np.float32)
    labels = np.asarray(y_list, dtype=np.int64)
    persons = np.asarray(person_list, dtype=str)
    heads = np.asarray(head_list, dtype=str)
    files = np.asarray(file_list, dtype=str)

    radar_feature_names = [
        "range", "doppler", "sin_h", "cos_h", "sin_v", "cos_v",
        "delta_range", "delta_doppler", "delta_sin_h", "delta_cos_h", "delta_sin_v", "delta_cos_v"
    ]
    imu_feature_names = [
        "qw", "qx", "qy", "qz", "qrel_w", "qrel_x", "qrel_y", "qrel_z",
        "dqw", "dqx", "dqy", "dqz"
    ]
    motion_feature_names = [
        "x_rel", "y_rel", "z_rel", "dx", "dy", "dz", "speed_xyz",
        "doppler", "delta_doppler", "range_rel", "delta_range"
    ]

    np.savez_compressed(
        cache_path,
        radar=radar_arr,
        imu=imu_arr,
        motion=motion_arr,
        labels=labels,
        persons=persons,
        heads=heads,
        files=files,
    )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "data_root": data_root,
            "class_mode": class_mode,
            "steps": steps,
            "considered": considered,
            "loaded": len(labels),
            "skipped": skipped,
            "label_names": labels_allowed,
            "radar_feature_names": radar_feature_names,
            "imu_feature_names": imu_feature_names,
            "motion_feature_names": motion_feature_names,
        }, f, indent=2, ensure_ascii=False)
    print(f"[DATA] considered={considered} loaded={len(labels)} skipped={skipped} radar={radar_arr.shape} imu={imu_arr.shape} motion={motion_arr.shape}")
    print(f"[CACHE SAVE] {cache_path}")
    return CachedDataset(radar_arr, imu_arr, motion_arr, labels, persons, heads, files, labels_allowed,
                         radar_feature_names, imu_feature_names, motion_feature_names)


# -----------------------------------------------------------------------------
# Splits and protocols
# -----------------------------------------------------------------------------

def split_subjects(persons: np.ndarray, fold: int, split_seed: int = 2026) -> Tuple[List[str], List[str], List[str]]:
    uniq = np.unique(persons)
    rng = np.random.default_rng(split_seed)
    shuffled = uniq.copy()
    rng.shuffle(shuffled)
    # Rotate so that each fold exposes different test/val combinations. With 13 subjects,
    # the exact 7/3/3 split is retained with overlapping folds, as in repeated holdout.
    rolled = np.roll(shuffled, -fold * 2)
    test = rolled[:3].tolist()
    val = rolled[3:6].tolist()
    train = rolled[6:].tolist()
    return train, val, test


def sample_balanced(indices: np.ndarray, labels: np.ndarray, heads: np.ndarray, target_n: int, seed: int) -> np.ndarray:
    if len(indices) <= target_n:
        return indices
    rng = np.random.default_rng(seed)
    # First try balanced by label and head to preserve diversity.
    buckets: Dict[Tuple[int, str], List[int]] = {}
    for idx in indices:
        key = (int(labels[idx]), str(heads[idx]))
        buckets.setdefault(key, []).append(int(idx))
    keys = list(buckets.keys())
    for k in keys:
        rng.shuffle(buckets[k])
    selected = []
    ptr = {k: 0 for k in keys}
    while len(selected) < target_n and keys:
        rng.shuffle(keys)
        new_keys = []
        for k in keys:
            if ptr[k] < len(buckets[k]) and len(selected) < target_n:
                selected.append(buckets[k][ptr[k]])
                ptr[k] += 1
            if ptr[k] < len(buckets[k]):
                new_keys.append(k)
        keys = new_keys
    if len(selected) < target_n:
        rest = np.setdiff1d(indices, np.asarray(selected, dtype=np.int64), assume_unique=False)
        rng.shuffle(rest)
        selected.extend(rest[:target_n-len(selected)].tolist())
    out = np.asarray(selected[:target_n], dtype=np.int64)
    rng.shuffle(out)
    return out


def protocol_indices(ds: CachedDataset, protocol: str, fold: int, seed: int, label_fraction: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    train_p, val_p, test_p = split_subjects(ds.persons, fold)
    persons = ds.persons
    heads = ds.heads
    labels = ds.labels
    train_all = np.where(np.isin(persons, train_p))[0]
    val_all = np.where(np.isin(persons, val_p))[0]
    test_all = np.where(np.isin(persons, test_p))[0]
    train_none = train_all[heads[train_all] == "None"]
    train_non = train_all[heads[train_all] != "None"]

    if protocol == "none_only":
        train_idx = train_none
        val_idx = val_all[heads[val_all] == "None"]
    elif protocol == "none_only_allhead_val":
        train_idx = train_none
        val_idx = val_all
    elif protocol == "mixed_full":
        train_idx = train_all
        val_idx = val_all
    elif protocol == "mixed_budget":
        train_idx = sample_balanced(train_all, labels, heads, target_n=len(train_none), seed=seed + fold * 1009)
        val_idx = val_all
    elif protocol == "partial_mixed":
        rng = np.random.default_rng(seed + fold * 1291)
        n_non = int(round(len(train_non) * float(label_fraction)))
        if n_non > 0:
            non_sel = sample_balanced(train_non, labels, heads, n_non, seed=seed + fold * 313)
            train_idx = np.concatenate([train_none, non_sel])
        else:
            train_idx = train_none
        rng.shuffle(train_idx)
        val_idx = val_all
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    meta = {
        "train_persons": train_p,
        "val_persons": val_p,
        "test_persons": test_p,
        "num_train_all": int(len(train_all)),
        "num_train_none": int(len(train_none)),
        "num_train_nonstatic": int(len(train_non)),
    }
    return train_idx.astype(np.int64), val_idx.astype(np.int64), test_all.astype(np.int64), meta


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= int(t) < n_classes and 0 <= int(p) < n_classes:
            cm[int(t), int(p)] += 1
    return cm


def metrics_from_cm(cm: np.ndarray) -> Dict[str, float]:
    n_classes = cm.shape[0]
    total = cm.sum()
    acc = float(np.trace(cm) / total) if total > 0 else 0.0
    precisions, recalls, f1s, supports = [], [], [], []
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        support = cm[c, :].sum()
        p = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        r = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        precisions.append(p); recalls.append(r); f1s.append(f); supports.append(support)
    supports_arr = np.asarray(supports, dtype=np.float64)
    f1_arr = np.asarray(f1s, dtype=np.float64)
    macro_f1 = float(f1_arr.mean()) if n_classes else 0.0
    weighted_f1 = float((f1_arr * supports_arr).sum() / supports_arr.sum()) if supports_arr.sum() > 0 else 0.0
    out = {"accuracy": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1}
    for c in range(n_classes):
        out[f"precision_{c}"] = precisions[c]
        out[f"recall_{c}"] = recalls[c]
        out[f"f1_{c}"] = f1s[c]
        out[f"support_{c}"] = float(supports[c])
    return out


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, heads: np.ndarray, label_names: List[str]) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    n_classes = len(label_names)
    cm = confusion_matrix(y_true, y_pred, n_classes)
    main = metrics_from_cm(cm)
    moving_mask = heads != "None"
    if moving_mask.any():
        cm_m = confusion_matrix(y_true[moving_mask], y_pred[moving_mask], n_classes)
        mm = metrics_from_cm(cm_m)
        main["moving_accuracy"] = mm["accuracy"]
        main["moving_macro_f1"] = mm["macro_f1"]
        main["moving_weighted_f1"] = mm["weighted_f1"]
    else:
        main["moving_accuracy"] = 0.0; main["moving_macro_f1"] = 0.0; main["moving_weighted_f1"] = 0.0
    # FAR for 5-class if 05None exists
    if "05None" in label_names:
        none_id = label_names.index("05None")
        recall_none = main.get(f"recall_{none_id}", 0.0)
        main["far"] = 1.0 - recall_none
        if moving_mask.any():
            cm_m = confusion_matrix(y_true[moving_mask], y_pred[moving_mask], n_classes)
            mm = metrics_from_cm(cm_m)
            main["moving_far"] = 1.0 - mm.get(f"recall_{none_id}", 0.0)
        else:
            main["moving_far"] = 0.0
    else:
        main["far"] = np.nan; main["moving_far"] = np.nan

    head_rows = []
    for h in CANON_HEADS:
        m = heads == h
        if not m.any():
            continue
        hm = metrics_from_cm(confusion_matrix(y_true[m], y_pred[m], n_classes))
        row = {"head": h, "num_samples": int(m.sum()), **hm}
        if "05None" in label_names:
            none_id = label_names.index("05None")
            row["far"] = 1.0 - hm.get(f"recall_{none_id}", 0.0)
        head_rows.append(row)
    head_df = pd.DataFrame(head_rows)

    class_rows = []
    for c, name in enumerate(label_names):
        class_rows.append({
            "class_id": c,
            "class_name": name,
            "precision": main.get(f"precision_{c}", 0.0),
            "recall": main.get(f"recall_{c}", 0.0),
            "f1": main.get(f"f1_{c}", 0.0),
            "support": main.get(f"support_{c}", 0.0),
        })
    class_df = pd.DataFrame(class_rows)
    return main, head_df, class_df


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

class TCNEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64, dropout: float = 0.15):
        super().__init__()
        self.proj = nn.Conv1d(input_dim, hidden, kernel_size=1)
        self.blocks = nn.ModuleList()
        for dilation in [1, 2, 4, 8]:
            self.blocks.append(nn.Sequential(
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=dilation, dilation=dilation),
                nn.BatchNorm1d(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            ))
        self.out_dim = hidden * 2
        self.seq_dim = hidden

    def forward_seq(self, x: torch.Tensor) -> torch.Tensor:
        # x: B,T,C -> B,T,H
        y = self.proj(x.transpose(1, 2))
        for block in self.blocks:
            y = y + block(y)
        return y.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.forward_seq(x)
        pooled = torch.cat([seq.mean(dim=1), seq.amax(dim=1)], dim=-1)
        return seq, pooled

class GRUEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64, dropout: float = 0.15):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.gru = nn.GRU(hidden, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.out_dim = hidden * 4
        self.seq_dim = hidden * 2

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.inp(x)
        seq, _ = self.gru(z)
        seq = self.drop(seq)
        pooled = torch.cat([seq.mean(dim=1), seq.amax(dim=1)], dim=-1)
        return seq, pooled

class TransformerSeqEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64, dropout: float = 0.15, max_len: int = 64):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden)
        self.pos = nn.Parameter(torch.zeros(1, max_len, hidden))
        layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=4, dim_feedforward=hidden*3,
                                           dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=2)
        self.out_dim = hidden * 2
        self.seq_dim = hidden
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.proj(x)
        z = z + self.pos[:, :z.size(1), :]
        seq = self.enc(z)
        pooled = torch.cat([seq.mean(dim=1), seq.amax(dim=1)], dim=-1)
        return seq, pooled


def build_encoder(backbone: str, input_dim: int, hidden: int, dropout: float, max_len: int) -> nn.Module:
    if backbone == "tcn":
        return TCNEncoder(input_dim, hidden, dropout)
    if backbone == "gru":
        return GRUEncoder(input_dim, hidden, dropout)
    if backbone == "transformer":
        return TransformerSeqEncoder(input_dim, hidden, dropout, max_len=max_len)
    raise ValueError(f"Unknown backbone: {backbone}")

class SingleStreamClassifier(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, backbone: str, hidden: int, dropout: float, max_len: int):
        super().__init__()
        self.encoder = build_encoder(backbone, input_dim, hidden, dropout, max_len)
        self.fc = nn.Sequential(
            nn.LayerNorm(self.encoder.out_dim),
            nn.Dropout(dropout),
            nn.Linear(self.encoder.out_dim, n_classes)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, pooled = self.encoder(x)
        return self.fc(pooled)

class LateFusionClassifier(nn.Module):
    def __init__(self, radar_dim: int, imu_dim: int, n_classes: int, backbone: str, hidden: int, dropout: float, max_len: int):
        super().__init__()
        self.radar_enc = build_encoder(backbone, radar_dim, hidden, dropout, max_len)
        self.imu_enc = build_encoder(backbone, imu_dim, hidden, dropout, max_len)
        dim = self.radar_enc.out_dim + self.imu_enc.out_dim
        self.fc = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(dropout), nn.Linear(dim, n_classes))
    def forward(self, radar: torch.Tensor, imu: torch.Tensor) -> torch.Tensor:
        _, rp = self.radar_enc(radar)
        _, ip = self.imu_enc(imu)
        return self.fc(torch.cat([rp, ip], dim=-1))

class AttentionFusionClassifier(nn.Module):
    def __init__(self, radar_dim: int, imu_dim: int, n_classes: int, backbone: str, hidden: int, dropout: float, max_len: int):
        super().__init__()
        self.radar_enc = build_encoder(backbone, radar_dim, hidden, dropout, max_len)
        self.imu_enc = build_encoder(backbone, imu_dim, hidden, dropout, max_len)
        rdim = self.radar_enc.seq_dim
        idim = self.imu_enc.seq_dim
        attn_dim = min(64, hidden)
        self.r_proj = nn.Linear(rdim, attn_dim)
        self.i_proj = nn.Linear(idim, attn_dim)
        self.mha = nn.MultiheadAttention(attn_dim, num_heads=4, dropout=dropout, batch_first=True)
        dim = self.radar_enc.out_dim + attn_dim * 2
        self.fc = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(dropout), nn.Linear(dim, n_classes))
    def forward(self, radar: torch.Tensor, imu: torch.Tensor) -> torch.Tensor:
        rseq, rp = self.radar_enc(radar)
        iseq, _ = self.imu_enc(imu)
        q = self.r_proj(rseq)
        kv = self.i_proj(iseq)
        attn_out, _ = self.mha(q, kv, kv, need_weights=False)
        ap = torch.cat([attn_out.mean(dim=1), attn_out.amax(dim=1)], dim=-1)
        return self.fc(torch.cat([rp, ap], dim=-1))


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# -----------------------------------------------------------------------------
# Experiment tensors and training
# -----------------------------------------------------------------------------

@dataclass
class PreparedData:
    train_inputs: Any
    val_inputs: Any
    test_inputs: Any
    y_train: torch.Tensor
    y_val: torch.Tensor
    y_test: torch.Tensor
    heads_val: np.ndarray
    heads_test: np.ndarray
    files_test: np.ndarray
    persons_test: np.ndarray
    input_dims: Dict[str, int]


def compute_norm(arr: np.ndarray, idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    flat = arr[idx].reshape(-1, arr.shape[-1])
    mean = flat.mean(axis=0, keepdims=True)
    std = flat.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_arr(arr: np.ndarray, mean: np.ndarray, std: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return ((arr[idx] - mean) / std).astype(np.float32)


def prepare_inputs(ds: CachedDataset, fusion_method: str, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray, device: torch.device) -> PreparedData:
    # Normalize each sensor stream using training subjects only.
    radar_mean, radar_std = compute_norm(ds.radar, train_idx)
    imu_mean, imu_std = compute_norm(ds.imu, train_idx)
    motion_mean, motion_std = compute_norm(ds.motion, train_idx)

    def to_t(x: np.ndarray) -> torch.Tensor:
        return torch.tensor(x, dtype=torch.float32, device=device)

    y_train = torch.tensor(ds.labels[train_idx], dtype=torch.long, device=device)
    y_val = torch.tensor(ds.labels[val_idx], dtype=torch.long, device=device)
    y_test = torch.tensor(ds.labels[test_idx], dtype=torch.long, device=device)

    if fusion_method == "radar_only":
        Xtr = to_t(normalize_arr(ds.radar, radar_mean, radar_std, train_idx))
        Xva = to_t(normalize_arr(ds.radar, radar_mean, radar_std, val_idx))
        Xte = to_t(normalize_arr(ds.radar, radar_mean, radar_std, test_idx))
        inputs = {"train": Xtr, "val": Xva, "test": Xte}
        dims = {"single": Xtr.shape[-1]}
    elif fusion_method == "early_fusion":
        rtr = normalize_arr(ds.radar, radar_mean, radar_std, train_idx)
        itr = normalize_arr(ds.imu, imu_mean, imu_std, train_idx)
        rva = normalize_arr(ds.radar, radar_mean, radar_std, val_idx)
        iva = normalize_arr(ds.imu, imu_mean, imu_std, val_idx)
        rte = normalize_arr(ds.radar, radar_mean, radar_std, test_idx)
        ite = normalize_arr(ds.imu, imu_mean, imu_std, test_idx)
        Xtr = to_t(np.concatenate([rtr, itr], axis=-1))
        Xva = to_t(np.concatenate([rva, iva], axis=-1))
        Xte = to_t(np.concatenate([rte, ite], axis=-1))
        inputs = {"train": Xtr, "val": Xva, "test": Xte}
        dims = {"single": Xtr.shape[-1]}
    elif fusion_method == "motion_domain_proposed":
        Xtr = to_t(normalize_arr(ds.motion, motion_mean, motion_std, train_idx))
        Xva = to_t(normalize_arr(ds.motion, motion_mean, motion_std, val_idx))
        Xte = to_t(normalize_arr(ds.motion, motion_mean, motion_std, test_idx))
        inputs = {"train": Xtr, "val": Xva, "test": Xte}
        dims = {"single": Xtr.shape[-1]}
    elif fusion_method in ["late_fusion", "attention_fusion"]:
        rtr = to_t(normalize_arr(ds.radar, radar_mean, radar_std, train_idx))
        itr = to_t(normalize_arr(ds.imu, imu_mean, imu_std, train_idx))
        rva = to_t(normalize_arr(ds.radar, radar_mean, radar_std, val_idx))
        iva = to_t(normalize_arr(ds.imu, imu_mean, imu_std, val_idx))
        rte = to_t(normalize_arr(ds.radar, radar_mean, radar_std, test_idx))
        ite = to_t(normalize_arr(ds.imu, imu_mean, imu_std, test_idx))
        inputs = {"train": (rtr, itr), "val": (rva, iva), "test": (rte, ite)}
        dims = {"radar": rtr.shape[-1], "imu": itr.shape[-1]}
    else:
        raise ValueError(f"Unknown fusion_method: {fusion_method}")

    return PreparedData(inputs["train"], inputs["val"], inputs["test"],
                        y_train, y_val, y_test,
                        ds.heads[val_idx], ds.heads[test_idx], ds.files[test_idx], ds.persons[test_idx], dims)


def build_model_for_experiment(fusion_method: str, backbone: str, input_dims: Dict[str, int], n_classes: int, hidden: int, dropout: float, max_len: int) -> nn.Module:
    if fusion_method in ["radar_only", "early_fusion", "motion_domain_proposed"]:
        return SingleStreamClassifier(input_dims["single"], n_classes, backbone, hidden, dropout, max_len)
    if fusion_method == "late_fusion":
        return LateFusionClassifier(input_dims["radar"], input_dims["imu"], n_classes, backbone, hidden, dropout, max_len)
    if fusion_method == "attention_fusion":
        return AttentionFusionClassifier(input_dims["radar"], input_dims["imu"], n_classes, backbone, hidden, dropout, max_len)
    raise ValueError(f"Unknown fusion_method: {fusion_method}")


def forward_model(model: nn.Module, inputs: Any, indices: torch.Tensor) -> torch.Tensor:
    if isinstance(inputs, tuple):
        return model(inputs[0][indices], inputs[1][indices])
    return model(inputs[indices])


def predict_all(model: nn.Module, inputs: Any, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    n = inputs[0].shape[0] if isinstance(inputs, tuple) else inputs.shape[0]
    preds = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            idx = torch.arange(start, min(start + batch_size, n), device=(inputs[0].device if isinstance(inputs, tuple) else inputs.device))
            logits = forward_model(model, inputs, idx)
            preds.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def evaluate_model(model: nn.Module, inputs: Any, y: torch.Tensor, heads: np.ndarray, label_names: List[str]) -> Dict[str, float]:
    pred = predict_all(model, inputs)
    y_np = y.detach().cpu().numpy()
    main, _, _ = compute_metrics(y_np, pred, heads, label_names)
    return main


def class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    weights = np.ones(n_classes, dtype=np.float32)
    nonzero = counts > 0
    if nonzero.any():
        weights[nonzero] = counts[nonzero].sum() / (nonzero.sum() * counts[nonzero])
    weights[~nonzero] = 0.0
    return torch.tensor(weights, dtype=torch.float32)


def train_one(prep: PreparedData, ds: CachedDataset, exp_cfg: Dict[str, Any], run_dir: str, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    n_classes = len(ds.label_names)
    model = build_model_for_experiment(
        exp_cfg["fusion_method"], exp_cfg["backbone"], prep.input_dims, n_classes,
        hidden=int(exp_cfg.get("hidden", 64)), dropout=float(exp_cfg.get("dropout", 0.15)),
        max_len=int(exp_cfg.get("num_steps", 40))
    ).to(device)

    cw = class_weights(prep.y_train.detach().cpu().numpy(), n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(exp_cfg.get("lr", 5e-4)), weight_decay=float(exp_cfg.get("weight_decay", 1e-4)))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=int(exp_cfg.get("scheduler_patience", 6)))
    batch_size = int(exp_cfg.get("batch_size", 512))
    epochs = int(exp_cfg.get("epochs", 80))
    patience = int(exp_cfg.get("patience", 12))
    eval_interval = int(exp_cfg.get("eval_interval", 2))
    augment = bool(exp_cfg.get("augment", True))
    jitter_std = float(exp_cfg.get("jitter_std", 0.01))
    scale_std = float(exp_cfg.get("scale_std", 0.03))

    best_state = None
    best_metric = -1.0
    best_epoch = 0
    no_improve = 0
    history = []
    n_train = len(prep.y_train)
    dev = device

    def maybe_augment(x: torch.Tensor) -> torch.Tensor:
        if not augment or not model.training:
            return x
        if jitter_std > 0:
            x = x + torch.randn_like(x) * jitter_std
        if scale_std > 0:
            scale = 1.0 + torch.randn((x.shape[0], 1, x.shape[-1]), device=x.device) * scale_std
            x = x * scale
        return x

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=dev)
        total_loss = 0.0
        correct = 0
        total = 0
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            if isinstance(prep.train_inputs, tuple):
                r = maybe_augment(prep.train_inputs[0][idx])
                im = maybe_augment(prep.train_inputs[1][idx])
                logits = model(r, im)
            else:
                x = maybe_augment(prep.train_inputs[idx])
                logits = model(x)
            loss = criterion(logits, prep.y_train[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(idx)
            correct += int((logits.argmax(dim=1) == prep.y_train[idx]).sum().item())
            total += len(idx)

        train_acc = correct / max(total, 1)
        train_loss = total_loss / max(total, 1)
        if epoch % eval_interval == 0 or epoch == 1 or epoch == epochs:
            val_metrics = evaluate_model(model, prep.val_inputs, prep.y_val, prep.heads_val, ds.label_names)
            select = val_metrics["macro_f1"]
            scheduler.step(select)
            improved = select > best_metric
            if improved:
                best_metric = select
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_moving_macro_f1": val_metrics.get("moving_macro_f1", 0.0),
                "lr": optimizer.param_groups[0]["lr"],
                "best": int(improved),
            })
            print(f"[E{epoch:03d}/{epochs}] loss={train_loss:.4f} tr_acc={train_acc:.4f} val_f1={val_metrics['macro_f1']:.4f} mov_f1={val_metrics.get('moving_macro_f1',0):.4f} lr={optimizer.param_groups[0]['lr']:.2e}")
            if no_improve >= patience:
                print(f"[EARLY STOP] epoch={epoch} best_epoch={best_epoch} best_val_f1={best_metric:.4f}")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(run_dir, "training_history.csv"), index=False)
    info = {"best_epoch": best_epoch, "best_val_macro_f1": best_metric, "num_params": count_params(model)}
    return model, info


# -----------------------------------------------------------------------------
# Run orchestration
# -----------------------------------------------------------------------------

def summarize_counts(name: str, ds: CachedDataset, idx: np.ndarray) -> None:
    labels = {ds.label_names[i]: int((ds.labels[idx] == i).sum()) for i in range(len(ds.label_names))}
    heads = {h: int((ds.heads[idx] == h).sum()) for h in CANON_HEADS if int((ds.heads[idx] == h).sum()) > 0}
    persons = len(np.unique(ds.persons[idx]))
    print(f"[{name}] n={len(idx)} persons={persons}")
    print(f"  labels: {labels}")
    print(f"  heads : {heads}")


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def run_experiment(data_root: str, result_root: str, exp_cfg: Dict[str, Any], fold: int, seed: int, args: argparse.Namespace) -> Dict[str, Any]:
    exp_id = exp_cfg["id"]
    class_mode = int(exp_cfg.get("class_mode", 4))
    run_dir = os.path.join(result_root, "runs", exp_id, f"fold_{fold:02d}", f"seed_{seed}")
    ensure_dir(run_dir)
    summary_path = os.path.join(run_dir, "run_summary.json")
    if args.resume and os.path.isfile(summary_path):
        print(f"[RESUME SKIP] {exp_id} fold={fold} seed={seed}")
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print("\n" + "="*100)
    print(f"RUN {exp_id} | fold={fold} | seed={seed} | fusion={exp_cfg['fusion_method']} | backbone={exp_cfg['backbone']} | protocol={exp_cfg['protocol']} | class={class_mode}")
    print("="*100)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ds = build_or_load_dataset(data_root, result_root, class_mode, int(exp_cfg.get("num_steps", 40)), force_rebuild=args.rebuild_cache)
    train_idx, val_idx, test_idx, split_meta = protocol_indices(ds, exp_cfg["protocol"], fold, seed, float(exp_cfg.get("label_fraction", 0.0)))
    summarize_counts("TRAIN", ds, train_idx)
    summarize_counts("VAL", ds, val_idx)
    summarize_counts("TEST", ds, test_idx)

    prep = prepare_inputs(ds, exp_cfg["fusion_method"], train_idx, val_idx, test_idx, device)
    model, train_info = train_one(prep, ds, exp_cfg, run_dir, device)

    # Predictions and metrics.
    y_test_np = prep.y_test.detach().cpu().numpy()
    pred_test = predict_all(model, prep.test_inputs)
    main, head_df, class_df = compute_metrics(y_test_np, pred_test, prep.heads_test, ds.label_names)
    head_df.to_csv(os.path.join(run_dir, "test_metrics_by_head.csv"), index=False)
    class_df.to_csv(os.path.join(run_dir, "test_metrics_by_class.csv"), index=False)

    pred_df = pd.DataFrame({
        "file": prep.files_test,
        "person": prep.persons_test,
        "head": prep.heads_test,
        "true_id": y_test_np,
        "true_name": [ds.label_names[i] for i in y_test_np],
        "pred_id": pred_test,
        "pred_name": [ds.label_names[i] for i in pred_test],
        "correct": (y_test_np == pred_test).astype(int),
    })
    if bool(exp_cfg.get("save_predictions", False)):
        pred_df.to_csv(os.path.join(run_dir, "test_predictions.csv"), index=False)

    summary = {
        "exp_id": exp_id,
        "fold": fold,
        "seed": seed,
        "class_mode": class_mode,
        "protocol": exp_cfg["protocol"],
        "label_fraction": float(exp_cfg.get("label_fraction", 0.0)),
        "fusion_method": exp_cfg["fusion_method"],
        "backbone": exp_cfg["backbone"],
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "num_test": int(len(test_idx)),
        "train_persons": split_meta["train_persons"],
        "val_persons": split_meta["val_persons"],
        "test_persons": split_meta["test_persons"],
        "input_dims": prep.input_dims,
        **train_info,
        **{f"test_{k}": float(v) for k, v in main.items() if isinstance(v, (float, int, np.floating))},
    }
    save_json(summary_path, summary)
    pd.DataFrame([summary]).to_csv(os.path.join(run_dir, "run_summary.csv"), index=False)
    torch.save({"model_state_dict": model.state_dict(), "summary": summary, "exp_cfg": exp_cfg}, os.path.join(run_dir, "model.pt"))
    print(f"[RUN DONE] {exp_id} fold={fold} seed={seed} acc={summary['test_accuracy']:.4f} macro_f1={summary['test_macro_f1']:.4f} moving_f1={summary['test_moving_macro_f1']:.4f}")
    return summary


def expand_manifest(manifest: Dict[str, Any]) -> List[Tuple[Dict[str, Any], int, int]]:
    tasks = []
    defaults = manifest.get("defaults", {})
    for exp in manifest.get("experiments", []):
        cfg = {**defaults, **exp}
        folds = cfg.get("folds", manifest.get("folds", [0, 1, 2, 3, 4]))
        if isinstance(folds, int):
            folds = list(range(folds))
        seeds = cfg.get("seeds", manifest.get("seeds", [42, 43, 44]))
        for fold in folds:
            for seed in seeds:
                tasks.append((cfg, int(fold), int(seed)))
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--result_root", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--show_pending", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.result_root)
    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    tasks = expand_manifest(manifest)
    print(f"[MANIFEST] {args.manifest}")
    print(f"[TASKS] {len(tasks)} runs")

    if args.show_pending:
        done = 0
        pending = []
        for cfg, fold, seed in tasks:
            summary_path = os.path.join(args.result_root, "runs", cfg["id"], f"fold_{fold:02d}", f"seed_{seed}", "run_summary.json")
            if os.path.isfile(summary_path):
                done += 1
            else:
                pending.append((cfg["id"], fold, seed, cfg["fusion_method"], cfg["backbone"], cfg["protocol"]))
        print(f"done={done} pending={len(pending)}")
        for row in pending[:200]:
            print("PENDING", row)
        if len(pending) > 200:
            print(f"... {len(pending)-200} more")
        return

    all_summaries = []
    for cfg, fold, seed in tasks:
        try:
            all_summaries.append(run_experiment(args.data_root, args.result_root, cfg, fold, seed, args))
        except Exception as e:
            run_dir = os.path.join(args.result_root, "runs", cfg["id"], f"fold_{fold:02d}", f"seed_{seed}")
            ensure_dir(run_dir)
            with open(os.path.join(run_dir, "FAILED.txt"), "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())
            print(f"[FAILED] {cfg['id']} fold={fold} seed={seed}: {e}")
            traceback.print_exc()
            if not manifest.get("continue_on_error", False):
                raise
    pd.DataFrame(all_summaries).to_csv(os.path.join(args.result_root, "last_manifest_runs.csv"), index=False)
    print("[DONE] manifest completed")

if __name__ == "__main__":
    main()
