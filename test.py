import os, copy, math, json, argparse, random, time, tracemalloc
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
from scipy.fft import fft, fftfreq

# ================================================================
# Optional: VMD import (pip install vmdpy)
# ================================================================
try:
    from vmdpy import VMD
except Exception:
    VMD = None

def apply_vmd(signal: np.ndarray, K: int = 3, alpha=2000, tau=0., DC=0, init=1, tol=1e-7):
    """Apply VMD to a 1D signal, return K modes shaped [K, T]."""
    if VMD is None:
        raise ImportError("vmdpy not installed. Install with: pip install vmdpy")
    u, _, _ = VMD(signal, alpha, tau, K, DC, init, tol)
    return u  # [K, T]

# ================================================================
# Utils
# ================================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_device(x, device):
    if isinstance(x, (list, tuple)):
        return [to_device(xx, device) for xx in x]
    return x.to(device)

# ================================================================
# Adaptive adjacency + graph conv
# ================================================================
class AdaptiveAdjacency(nn.Module):
    def __init__(self, num_nodes: int, emb_dim: int, ablate: bool = False):
        super().__init__()
        self.ablate = ablate
        if not self.ablate:
            self.E1 = nn.Parameter(torch.randn(num_nodes, emb_dim) * 0.1)
            self.E2 = nn.Parameter(torch.randn(num_nodes, emb_dim) * 0.1)
        self.num_nodes = num_nodes

    def forward(self):
        if self.ablate:
            # Return identity matrix for ablation study
            return torch.eye(self.num_nodes, device=self.E1.device if hasattr(self, 'E1') else 'cpu')
        logits = F.relu(self.E1 @ self.E2.t())  # [N,N]
        return F.softmax(logits, dim=-1)

class GraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # x: [B,N,C], A: [N,N]
        x_mp = torch.einsum("ij,bjc->bic", A, x)
        return self.lin(x_mp)

# ================================================================
# GraphConvLSTM cell
# ================================================================
class GraphConvLSTMCell(nn.Module):
    def __init__(self, num_nodes: int, in_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gc_x = GraphConv(in_dim, 4 * hidden_dim)
        self.gc_h = GraphConv(hidden_dim, 4 * hidden_dim)

    def forward(self, x_t, h_prev, c_prev, A):
        gates = self.gc_x(x_t, A) + self.gc_h(h_prev, A)  # [B,N,4H]
        i, f, o, g = torch.chunk(gates, 4, dim=-1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)
        return h_t, c_t

# ================================================================
# LSTM-DSTGCRN
# ================================================================
class LSTMDSTGCRN(nn.Module):
    """
    x: [B,T_in,N,C_in] -> y: [B,T_out,N,C_out]
    If using VMD and predicting IMFs: C_out = K (number of modes). Otherwise C_out = 1.
    """
    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        horizon: int = 3,
        num_layers: int = 1,
        emb_dim: int = 16,
        use_attention: bool = False,
        num_heads: int = 2,
        ablate_adja: bool = False,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon
        self.adaptiveA = AdaptiveAdjacency(num_nodes, emb_dim, ablate=ablate_adja)
        self.use_attention = use_attention
        if use_attention:
            assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
            # Using hidden_dim for attention, as input is projected to hidden_dim first
            self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
            self.input_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.attn = None

        cells = []
        for l in range(num_layers):
            in_dim = hidden_dim if self.use_attention else (input_dim if l == 0 else hidden_dim)
            cells.append(GraphConvLSTMCell(num_nodes, in_dim, hidden_dim))
        self.cells = nn.ModuleList(cells)

        self.proj = nn.Linear(hidden_dim, output_dim * horizon)

        # expose groups for selective integration
        self.modules_to_integrate = {
            "adaptiveA": self.adaptiveA,
            "cells": self.cells,
            "proj": self.proj,
        }
        if self.attn is not None:
            self.modules_to_integrate["attn"] = self.attn
            self.modules_to_integrate["input_proj"] = self.input_proj

    def forward(self, x):
        # x: [B,T,N,C]
        B, T, N, C = x.shape
        assert N == self.num_nodes, f"Input N={N} but model expects {self.num_nodes}"
        A = self.adaptiveA()  # [N,N]

        if self.attn is not None:
            x = self.input_proj(x) # Project to hidden_dim
            x_flat = x.permute(0, 2, 1, 3).reshape(B * N, T, -1)
            x_attn, _ = self.attn(x_flat, x_flat, x_flat)
            x = x_attn.reshape(B, N, T, -1).permute(0, 2, 1, 3)

        hs = [x.new_zeros(B, N, c.hidden_dim) for c in self.cells]
        cs = [x.new_zeros(B, N, c.hidden_dim) for c in self.cells]

        for t in range(T):
            xt = x[:, t]
            for i, cell in enumerate(self.cells):
                h, c = cell(xt, hs[i], cs[i], A)
                hs[i], cs[i] = h, c
                xt = h

        out = self.proj(hs[-1])  # [B,N, C_out*horizon]
        out = out.view(B, N, self.horizon, -1).permute(0, 2, 1, 3)  # [B,T_out,N,C_out]
        return out

# ================================================================
# Dataset (with optional VMD preprocessing)
# ================================================================
class TripWeatherDataset(Dataset):
    """
    Builds features per the paper. Two modes:
      - VMD disabled (vmd_k=0): features = [trips, temperature, precipitation, hour_norm, day_norm, weekend]
      - VMD enabled (vmd_k=K):  features = [IMF_1..IMF_K, temperature, precipitation, hour_norm, day_norm, weekend]

    Targets always return both:
      - y_imfs: IMF stack (when K>0) or trips (when K=0)  -> used by model output
      - y_trips: raw trips (for reconstruction loss & metrics)
    """
    def __init__(
        self,
        trip_csv: str,
        input_len: int = 12,
        output_len: int = 3,
        stride: int = 1,
        node_subset: Optional[List[int]] = None,
        vmd_k: int = 0,
        save_vmd: bool = True,
    ):
        self.T_in = input_len
        self.T_out = output_len
        self.stride = stride
        self.node_subset = None if node_subset is None else list(node_subset)
        self.vmd_k = int(vmd_k)

        trips = pd.read_csv(trip_csv)
        weather_csv = trip_csv.replace("tripdata", "weatherdata")
        weathers = pd.read_csv(weather_csv)

        # parse timestamps & drop tz
        trips["timestamp"] = pd.to_datetime(trips["timestamp"]).dt.tz_localize(None)
        weathers["timestamp"] = pd.to_datetime(weathers["timestamp"]).dt.tz_localize(None)

        trips = trips.set_index("timestamp")
        weathers = weathers.set_index("timestamp")

        trips_np = trips.to_numpy()            # [T, N]
        weathers_np = weathers.to_numpy()      # [T, 2N]
        self.full_trip_series = trips_np # Store for MASE calculation

        # ---------- Time/aux features ----------
        weekends = trips.index.dayofweek.isin([5, 6]).astype(int)
        enc = OneHotEncoder(sparse_output=False)
        weekend_1hot = enc.fit_transform(weekends.reshape(-1, 1))[:, 0].reshape(-1, 1)
        weekend_1hot = np.repeat(weekend_1hot[:, np.newaxis, :], trips_np.shape[1], axis=1)  # [T,N,1]
        day_norm = (trips.index.dayofweek.to_numpy() / 6.0) # Monday=0, Sunday=6
        day_norm = np.repeat(day_norm[:, None], trips_np.shape[1], axis=1)
        hour_norm = (trips.index.hour.to_numpy() / 23.0) # 0 to 23
        hour_norm = np.repeat(hour_norm[:, None], trips_np.shape[1], axis=1)
        temperature = weathers_np[:, ::2]
        precipitation = weathers_np[:, 1::2]

        # ---------- Primary feature block (trips vs VMD IMFs) ----------
        if self.vmd_k > 0:
            # VMD decomposition per node
            imf_list = []
            print(f"Applying VMD (K={self.vmd_k}) to {trips_np.shape[1]} nodes...")
            for j in range(trips_np.shape[1]):
                modes = apply_vmd(trips_np[:, j], K=self.vmd_k)  # [K, T]
                imf_list.append(modes)
            imfs_stacked = np.stack(imf_list, axis=1).transpose(2, 1, 0)  # [T, N, K]

            # Save once per file (avoid duplicates across clients)
            vmd_path = trip_csv.replace("tripdata", f"tripdata_vmd{self.vmd_k}")
            if save_vmd and not os.path.exists(vmd_path):
                vmd_df = pd.DataFrame(imfs_stacked.reshape(imfs_stacked.shape[0], -1))
                vmd_df.to_csv(vmd_path, index=False)
                print(f"Saved VMD-preprocessed trips to {vmd_path}")

            primary = imfs_stacked  # [T,N,K]
            y_imfs_full = imfs_stacked  # [T,N,K]
        else:
            primary = trips_np[:, :, None]  # [T,N,1]
            y_imfs_full = primary  # treat trips as single-channel output

        # Build full feature tensor
        data = np.concatenate(
            (
                primary,
                temperature[:, :, None],
                precipitation[:, :, None],
                hour_norm[:, :, None],
                day_norm[:, :, None],
                weekend_1hot,
            ),
            axis=2,
        )  # [T,N, (K or 1) + 5]

        # Optionally subset nodes for this client
        if self.node_subset is not None:
            data_sub = data[:, self.node_subset, :]
            y_imfs_sub = y_imfs_full[:, self.node_subset, :]
            trips_sub = trips_np[:, self.node_subset][:, :, None]
        else:
            data_sub = data
            y_imfs_sub = y_imfs_full
            trips_sub = trips_np[:, :, None]

        self.data = torch.tensor(data_sub, dtype=torch.float32)      # [T,N,F]
        self.y_imfs = torch.tensor(y_imfs_sub, dtype=torch.float32)     # [T,N,K or 1]
        self.y_trips = torch.tensor(trips_sub, dtype=torch.float32)      # [T,N,1]

        # window indices
        Ttot = self.data.shape[0]
        self.windows = []
        for s in range(0, Ttot - (self.T_in + self.T_out) + 1, self.stride):
            x = self.data[s: s + self.T_in]                      # [T_in,N,F]
            y_imf = self.y_imfs[s + self.T_in: s + self.T_in + self.T_out]  # [T_out,N,K or 1]
            y_trip = self.y_trips[s + self.T_in: s + self.T_in + self.T_out] # [T_out,N,1]
            self.windows.append((x, y_imf, y_trip))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]

# ---------------------------------------------------------------
# Collate and padding helpers
# ---------------------------------------------------------------

def collate_batch(batch):
    xs, y_imfs, y_trips = zip(*batch)
    return torch.stack(xs, 0), (torch.stack(y_imfs, 0), torch.stack(y_trips, 0))


def pad_batch_to_full(x_sub: torch.Tensor, subset_idx: List[int], N_full: int):
    """
    x_sub: [B,T,N_sub,C]
    returns x_full: [B,T,N_full,C], mask: [B,T,N_full,1]
    """
    B, T, N_sub, C = x_sub.shape
    x_full = x_sub.new_zeros((B, T, N_full, C))
    mask = x_sub.new_zeros((B, T, N_full, 1))
    for i, pos in enumerate(subset_idx):
        x_full[:, :, pos, :] = x_sub[:, :, i, :]
        mask[:, :, pos, 0] = 1.0
    return x_full, mask

# ================================================================
# Train / Eval (with IMF→trip reconstruction for loss & metrics)
# ================================================================
def _mae(a, b, mask=None):
    if mask is None:
        return (a - b).abs().mean()
    diff = (a - b).abs() * mask
    return diff.sum() / mask.sum().clamp(min=1)

def _rmse(a, b, mask=None):
    if mask is None:
        return torch.sqrt(((a - b) ** 2).mean())
    diff2 = ((a - b) ** 2) * mask
    return torch.sqrt(diff2.sum() / mask.sum().clamp(min=1))

def _mape(a, b, mask=None, epsilon=1e-8):
    abs_err = (a - b).abs()
    abs_true = b.abs()
    
    # Calculate MAPE
    per_err = abs_err / abs_true.clamp(min=epsilon)
    
    if mask is not None:
        masked_per_err = per_err * mask
        return 100. * (masked_per_err.sum() / mask.sum().clamp(min=1))
    else:
        return 100. * per_err.mean()

def _mase(a, b, scale_factor, mask=None):
    if scale_factor < 1e-8: return float('inf')
    mae_val = _mae(a, b, mask)
    return mae_val / scale_factor

def _r2(a, b, mask=None):
    if mask is not None:
        mask_flat = mask.squeeze(-1).flatten().cpu().numpy() > 0.5
        a_masked = a.flatten().cpu().numpy()[mask_flat]
        b_masked = b.flatten().cpu().numpy()[mask_flat]
        if len(a_masked) < 2: # R2 is not well-defined for less than 2 points
            return 0.0
        return r2_score(b_masked, a_masked)
    return r2_score(b.cpu().numpy().flatten(), a.cpu().numpy().flatten())

def calculate_mase_scaling_factor(train_dataset: Dataset):
    """Calculates the MASE scaling factor from the training set (naive forecast error)."""
    # train_dataset.dataset is the underlying TripWeatherDataset
    full_series = train_dataset.dataset.full_trip_series
    train_indices = train_dataset.indices
    # Get the part of the series corresponding to the training set
    train_series = full_series[min(train_indices):max(train_indices)+1]
    
    # Naive forecast (y_t = y_{t-1})
    naive_forecast_error = np.abs(train_series[1:] - train_series[:-1])
    return np.mean(naive_forecast_error)

def train_one_epoch(model, loader, optimizer, device, subset_idx=None, N_full=None):
    model.train()
    total_mae = 0.0
    n_batches = 0

    for x, (y_imf, y_trips) in loader:
        # x: [B,T_in,N,F], y_imf: [B,T_out,N,Kor1], y_trips: [B,T_out,N,1]
        mask = None
        if subset_idx is not None:
            x, _ = pad_batch_to_full(x, subset_idx, N_full)
            y_imf, _ = pad_batch_to_full(y_imf, subset_idx, N_full)
            y_trips, mask = pad_batch_to_full(y_trips, subset_idx, N_full)
        
        x, y_imf, y_trips = to_device(x, device), to_device(y_imf, device), to_device(y_trips, device)
        if mask is not None: mask = to_device(mask, device)

        optimizer.zero_grad()
        y_hat_imf = model(x)                      # [B,T_out,N,Kor1]
        y_hat_trips = y_hat_imf.sum(dim=-1, keepdim=True)  # reconstruct trips

        loss = _mae(y_hat_trips, y_trips, mask=mask)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_mae += loss.item()
        n_batches += 1

    return total_mae / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, mase_scaler, subset_idx=None, N_full=None):
    model.eval()
    all_y_trips = []
    all_y_hat_trips = []
    all_masks = []
    
    for x, (y_imf, y_trips) in loader:
        mask = None
        if subset_idx is not None:
            x, _ = pad_batch_to_full(x, subset_idx, N_full)
            y_trips, mask = pad_batch_to_full(y_trips, subset_idx, N_full)
            all_masks.append(mask.cpu())
        else:
            all_masks.append(torch.ones_like(y_trips.cpu()))

        x, y_trips = to_device(x, device), to_device(y_trips, device)
        
        y_hat_imf = model(x)
        y_hat_trips = y_hat_imf.sum(dim=-1, keepdim=True)
        
        all_y_trips.append(y_trips.cpu())
        all_y_hat_trips.append(y_hat_trips.cpu())

    # Concatenate all results
    all_y_trips = torch.cat(all_y_trips, dim=0)
    all_y_hat_trips = torch.cat(all_y_hat_trips, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    # Calculate metrics on the full dataset
    mae = _mae(all_y_hat_trips, all_y_trips, mask=all_masks)
    rmse = _rmse(all_y_hat_trips, all_y_trips, mask=all_masks)
    mape = _mape(all_y_hat_trips, all_y_trips, mask=all_masks)
    mase = _mase(all_y_hat_trips, all_y_trips, mase_scaler, mask=all_masks)
    r2 = _r2(all_y_hat_trips, all_y_trips, mask=all_masks)
    
    return mae, rmse, mape, mase, r2, all_y_trips.numpy(), all_y_hat_trips.numpy()

# ================================================================
# Algorithm 2: selective integration (unchanged API)
# ================================================================

def _copy_group(dst: nn.Module, src: nn.Module, group_name: str):
    dst_group = dst.modules_to_integrate[group_name]
    src_group = src.modules_to_integrate[group_name]
    dst_group.load_state_dict(src_group.state_dict())


def selective_integration(local_model, global_model, val_loader, device, mase_scaler, subset_idx=None, N_full=None, groups=None):
    if groups is None:
        groups = list(local_model.modules_to_integrate.keys())

    best_loss = math.inf
    best_group = None
    base = copy.deepcopy(local_model)

    # First, evaluate the local model as a baseline
    base_mae, _, _, _, _, _, _ = evaluate(base, val_loader, device, mase_scaler, subset_idx=subset_idx, N_full=N_full)
    best_loss = base_mae

    for g in groups:
        cand = copy.deepcopy(base)
        _copy_group(cand, global_model, g)
        val_mae, _, _, _, _, _, _ = evaluate(cand, val_loader, device, mase_scaler, subset_idx=subset_idx, N_full=N_full)
        if val_mae < best_loss:
            best_loss, best_group = val_mae, g

    if best_group is not None:
        _copy_group(local_model, global_model, best_group)
    
    return local_model, best_group

# ================================================================
# Federated classes
# ================================================================
@dataclass
class ClientConfig:
    id: int
    epochs: int = 2
    batch_size: int = 32
    lr: float = 1e-3

class FLClient:
    def __init__(self, cid, model_fn, train_ds, val_ds, test_ds, cfg: ClientConfig, device, subset_idx: List[int], N_full: int):
        self.cid = cid
        self.device = device
        self.cfg = cfg
        self.model = model_fn().to(device)
        self.train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_batch)
        self.val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_batch)
        self.test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_batch)
        self.subset_idx = subset_idx
        self.N_full = N_full
        # Calculate MASE scaler once from this client's training data
        self.mase_scaler = calculate_mase_scaling_factor(train_ds)

    def set_params_from(self, global_model):
        self.model.load_state_dict(global_model.state_dict())

    def integrate(self, global_model):
        self.model, best_group = selective_integration(self.model, global_model, self.val_loader, self.device, self.mase_scaler, subset_idx=self.subset_idx, N_full=self.N_full)
        return best_group

    def train_local(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        local_loss_history = []
        for _ in range(self.cfg.epochs):
            loss = train_one_epoch(self.model, self.train_loader, opt, self.device, subset_idx=self.subset_idx, N_full=self.N_full)
            local_loss_history.append(loss)
        return local_loss_history

    def state_dict(self):
        return copy.deepcopy(self.model.state_dict())

    def n_train(self): return len(self.train_loader.dataset)
    def n_test(self):  return len(self.test_loader.dataset)
    def n_val(self): return len(self.val_loader.dataset)

    @torch.no_grad()
    def test_metrics(self):
        return evaluate(self.model, self.test_loader, self.device, self.mase_scaler, subset_idx=self.subset_idx, N_full=self.N_full)

    @torch.no_grad()
    def train_metrics(self):
        return evaluate(self.model, self.train_loader, self.device, self.mase_scaler, subset_idx=self.subset_idx, N_full=self.N_full)

    @torch.no_grad()
    def val_metrics(self):
        return evaluate(self.model, self.val_loader, self.device, self.mase_scaler, subset_idx=self.subset_idx, N_full=self.N_full)

class FLServer:
    def __init__(self, model_fn, clients: List[FLClient], N_full: int):
        self.global_model = model_fn()
        self.clients = clients
        self.N_full = N_full

    def distribute(self):
        for c in self.clients:
            c.set_params_from(self.global_model)

    def aggregate_fedavg(self, states: List[Tuple[Dict, int]]):
        total = sum(n for _, n in states)
        if total == 0: return
        new_sd = copy.deepcopy(states[0][0])
        for k in new_sd.keys():
            new_sd[k] = sum(sd[k] * (n/total) for sd, n in states)
        self.global_model.load_state_dict(new_sd)

# ================================================================
# Federated loop with global metrics + plotting
# ================================================================
def _plot_all_metrics(log, save_dir, dataset_name):
    rounds = [r["round"] for r in log]
    maes = [r["global_test_mae"] for r in log]
    rmses = [r["global_test_rmse"] for r in log]
    train_maes = [r["global_train_mae"] for r in log]
    train_rmses = [r["global_train_rmse"] for r in log]

    plt.figure(figsize=(10, 6))
    plt.plot(rounds, maes, 'o-', label="Test MAE")
    plt.plot(rounds, rmses, 's-', label="Test RMSE")
    plt.plot(rounds, train_maes, 'o--', label="Train MAE", alpha=0.7)
    plt.plot(rounds, train_rmses, 's--', label="Train RMSE", alpha=0.7)
    plt.xlabel("Round")
    plt.ylabel("Metric")
    plt.title(f"Performance Metrics over Rounds ({dataset_name})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    png_path = os.path.join(save_dir, f"{dataset_name}_metrics.png")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()
    print(f"Saved performance metrics plot to {png_path}")

def _plot_training_loss(log, save_dir, dataset_name):
    rounds = [r["round"] for r in log]
    loss = [r["global_train_loss_avg"] for r in log]
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, loss, 'o-', label="Average Client Training Loss (MAE)")
    plt.xlabel("Round")
    plt.ylabel("Training Loss (MAE)")
    plt.title(f"Federated Training Loss over Rounds ({dataset_name})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    png_path = os.path.join(save_dir, f"{dataset_name}_training_loss.png")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()
    print(f"Saved training loss plot to {png_path}")

def _plot_module_replacement(log, save_dir, dataset_name, module_names):
    rounds = [r["round"] for r in log]
    module_counts = {name: [] for name in module_names}
    module_counts["None"] = [] # For rounds where no module improved performance

    for r_log in log:
        counts = {name: 0 for name in module_names}
        counts["None"] = 0
        for group in r_log["module_replacement"]:
            if group is None:
                counts["None"] += 1
            else:
                counts[group] += 1
        for name in module_counts:
            module_counts[name].append(counts[name])

    plt.figure(figsize=(12, 7))
    bottom = np.zeros(len(rounds))
    for name, counts in module_counts.items():
        plt.bar(rounds, counts, label=name, bottom=bottom)
        bottom += np.array(counts)
    
    plt.xlabel("Round")
    plt.ylabel("Number of Clients")
    plt.title(f"Module Replacement Choices During FL ({dataset_name})")
    plt.legend(title="Module Chosen")
    plt.xticks(rounds)
    plt.tight_layout()
    png_path = os.path.join(save_dir, f"{dataset_name}_module_replacement.png")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()
    print(f"Saved module replacement plot to {png_path}")


def _plot_temporal_stability(y_true, y_pred, save_dir, dataset_name):
    # Flatten across nodes and horizon
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    abs_errors = np.abs(y_true_flat - y_pred_flat)
    
    # Calculate rolling MAE to smooth the plot
    window_size = 24 * 7 # 1 week
    df = pd.DataFrame({'error': abs_errors})
    rolling_mae = df['error'].rolling(window=window_size, min_periods=1, center=True).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(rolling_mae.index, rolling_mae.values, label=f'Rolling MAE ({window_size}-step window)')
    plt.xlabel('Time Step')
    plt.ylabel('MAE')
    plt.title(f'Temporal Stability Analysis ({dataset_name})')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    png_path = os.path.join(save_dir, f"{dataset_name}_temporal_stability.png")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()
    print(f"Saved temporal stability plot to {png_path}")

def _plot_true_vs_predicted(y_true, y_pred, save_dir, dataset_name):
    # Plot for the first node, first prediction horizon step
    plt.figure(figsize=(15, 7))
    sample_true = y_true[:, 0, 0, 0]
    sample_pred = y_pred[:, 0, 0, 0]
    time_steps = min(len(sample_true), 24 * 14) # Plot up to 2 weeks
    
    plt.plot(sample_true[:time_steps], label="True Demands", alpha=0.8)
    plt.plot(sample_pred[:time_steps], label="Predicted Demands", alpha=0.8, linestyle='--')
    plt.xlabel("Time Step (Hour)")
    plt.ylabel("Demand")
    plt.title(f"True vs. Predicted Demands for a Sample Node ({dataset_name})")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    png_path = os.path.join(save_dir, f"{dataset_name}_true_vs_predicted.png")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()
    print(f"Saved true vs predicted plot to {png_path}")

def _plot_error_distribution(y_true, y_pred, save_dir, dataset_name):
    errors = (y_pred - y_true).flatten()
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=100, density=True, label='Error Distribution')
    mean_err = np.mean(errors)
    std_err = np.std(errors)
    plt.axvline(mean_err, color='r', linestyle='--', label=f'Mean = {mean_err:.2f}')
    plt.xlabel("Error (Predicted - True)")
    plt.ylabel("Density")
    plt.title(f"Error Distribution ({dataset_name}) | Std Dev: {std_err:.2f}")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    png_path = os.path.join(save_dir, f"{dataset_name}_error_distribution.png")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()
    print(f"Saved error distribution plot to {png_path}")

def _plot_frequency_domain_analysis(signal, vmd_recon, model_pred, save_dir, dataset_name):
    # The input arrays are already 1D series for a single node.
    # No re-indexing is needed.

    vmd_error = signal - vmd_recon
    model_error = signal - model_pred

    N = len(signal)
    T = 1.0 # Assuming hourly data
    
    yf_vmd = fft(vmd_error)
    yf_model = fft(model_error)
    xf = fftfreq(N, T)[:N//2]
    
    plt.figure(figsize=(14, 7))
    plt.semilogy(xf, 2.0/N * np.abs(yf_vmd[0:N//2]), label='VMD Reconstruction Error Spectrum', alpha=0.7)
    plt.semilogy(xf, 2.0/N * np.abs(yf_model[0:N//2]), label='Final Model Prediction Error Spectrum', alpha=0.7)
    plt.xlabel('Frequency (cycles/hour)')
    plt.ylabel('Amplitude')
    plt.title(f'Frequency Domain Error Reduction ({dataset_name})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    png_path = os.path.join(save_dir, f"{dataset_name}_frequency_error.png")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()
    print(f"Saved frequency domain plot to {png_path}")


def _calculate_snr(signal, noise):
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    if noise_power < 1e-9: return float('inf')
    return 10 * np.log10(signal_power / noise_power)

def run_federated(
    trip_csv: str,
    dataset_name: str,
    num_clients: int = 3,
    num_rounds: int = 10,
    input_len: int = 12,
    output_len: int = 3,
    device: str = None,
    save_dir: str = "./ckpts",
    epochs_per_round: int = 2,
    batch_size: int = 16,
    use_attention: bool = False,
    attn_heads: int = 2,
    vmd_k: int = 0,
    save_vmd: bool = True,
    ablate_adja: bool = False,
):
    # --- Setup ---
    os.makedirs(save_dir, exist_ok=True)
    set_seed(42)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tracemalloc.start()
    start_time = time.time()

    # --- Infer N (nodes) ---
    tmp = pd.read_csv(trip_csv, nrows=1)
    node_cols = [c for c in tmp.columns if c != "timestamp"]
    N_full = len(node_cols)
    node_indices = list(range(N_full))

    # --- VMD Analysis ---
    vmd_results = {}
    vmd_reconstruction_for_plot = None
    original_signal_for_plot = None
    if vmd_k > 0:
        print("\n--- VMD analysis ---")
        tmp_ds = TripWeatherDataset(trip_csv, vmd_k=0, node_subset=node_indices[:1]) # just for one node
        signal = tmp_ds.full_trip_series[:, 0]
        imfs = apply_vmd(signal, K=vmd_k)
        reconstructed = imfs.sum(axis=0)
        
        reconstruction_snr = _calculate_snr(signal, signal - reconstructed)
        
        vmd_results = {"Reconstruction_SNR_dB": reconstruction_snr}
        print(f"VMD Reconstruction SNR: {reconstruction_snr:.2f} dB")
        json_path_vmd = os.path.join(save_dir, f"{dataset_name}_vmd_results.json")
        with open(json_path_vmd, "w") as f:
            json.dump(vmd_results, f, indent=2)
            
        original_signal_for_plot = signal
        vmd_reconstruction_for_plot = reconstructed


    # --- Client and Server Setup ---
    chunks = [node_indices[i::num_clients] for i in range(num_clients)]
    input_dim = (vmd_k if vmd_k > 0 else 1) + 5
    output_dim = (vmd_k if vmd_k > 0 else 1)
    
    def model_fn():
        return LSTMDSTGCRN(
            num_nodes=N_full, input_dim=input_dim, hidden_dim=64, output_dim=output_dim,
            horizon=output_len, use_attention=use_attention, num_heads=attn_heads, ablate_adja=ablate_adja
        )

    clients: List[FLClient] = []
    print("\n--- Initializing Datasets for Clients ---")
    for cid, subset in enumerate(chunks):
        full_ds = TripWeatherDataset(trip_csv, input_len, output_len, stride=1, node_subset=subset, vmd_k=vmd_k, save_vmd=save_vmd and cid == 0)
        n_total = len(full_ds)
        if n_total < 10: # Ensure enough data for split
            print(f"Warning: Client {cid} has only {n_total} samples. Skipping this client.")
            continue
        n_tr = int(0.70 * n_total)
        n_va = int(0.15 * n_total)
        n_te = n_total - n_tr - n_va
        tr, va, te = torch.utils.data.random_split(full_ds, [n_tr, n_va, n_te], generator=torch.Generator().manual_seed(123 + cid))
        cfg = ClientConfig(id=cid, epochs=epochs_per_round, batch_size=batch_size, lr=1e-3)
        clients.append(FLClient(cid, model_fn, tr, va, te, cfg, device, subset_idx=subset, N_full=N_full))

    if not clients:
        raise ValueError("No clients were created. Check data size and splitting logic.")
        
    server = FLServer(model_fn, clients, N_full=N_full)
    global_log = []

    # --- Federated Training Loop ---
    for r in range(num_rounds):
        print(f"\n===== Round {r+1}/{num_rounds} | {dataset_name} =====")
        round_start_time = time.time()
        
        server.distribute()
        
        client_train_losses = []
        best_groups_this_round = []
        for c in clients:
            best_group = c.integrate(server.global_model)
            best_groups_this_round.append(best_group)
            
            local_losses = c.train_local()
            client_train_losses.append(np.mean(local_losses))
        
        avg_client_loss = np.mean(client_train_losses)

        # Weighted global training metrics (on each client's TRAIN set)
        train_metrics = [(c.train_metrics()[:5], c.n_train()) for c in clients]
        total_train = sum(n for _, n in train_metrics)
        global_train_mae = sum(m[0] * n for (m), n in train_metrics) / total_train
        global_train_rmse = sum(m[1] * n for (m), n in train_metrics) / total_train

        # Evaluate test metrics per client and aggregate
        test_metrics_data = [c.test_metrics() for c in clients]
        total_test = sum(c.n_test() for c in clients)
        
        global_mae = sum(m[0] * c.n_test() for m, c in zip(test_metrics_data, clients)) / total_test
        global_rmse = sum(m[1] * c.n_test() for m, c in zip(test_metrics_data, clients)) / total_test
        global_mape = sum(m[2] * c.n_test() for m, c in zip(test_metrics_data, clients)) / total_test
        global_mase = sum(m[3] * c.n_test() for m, c in zip(test_metrics_data, clients)) / total_test
        global_r2 = sum(m[4] * c.n_test() for m, c in zip(test_metrics_data, clients)) / total_test
        
        print(f"Global Test MAE: {global_mae:.4f} | RMSE: {global_rmse:.4f} | R^2: {global_r2:.4f}")
        print(f"Avg Train Loss: {avg_client_loss:.4f} | Global Train MAE: {global_train_mae:.4f}")

        log_entry = {
            "round": r + 1, "global_train_loss_avg": float(avg_client_loss),
            "global_train_mae": float(global_train_mae), "global_train_rmse": float(global_train_rmse),
            "global_test_mae": float(global_mae), "global_test_rmse": float(global_rmse),
            "global_test_mape": float(global_mape), "global_test_mase": float(global_mase), "global_test_r2": float(global_r2),
            "module_replacement": best_groups_this_round
        }
        global_log.append(log_entry)

        states = [(c.state_dict(), c.n_train()) for c in clients]
        server.aggregate_fedavg(states)
        
        print(f"Round {r+1} took {time.time() - round_start_time:.2f} seconds.")
    
    # --- Final Evaluation and Reporting ---
    print("\n--- Final Evaluation and Plotting ---")
    server.distribute()
    
    final_y_true_list, final_y_pred_list = [], []
    for c in clients:
        _, _, _, _, _, y_true, y_pred = c.test_metrics()
        final_y_true_list.append(y_true)
        final_y_pred_list.append(y_pred)
    
    final_y_true = np.concatenate(final_y_true_list, axis=0)
    final_y_pred = np.concatenate(final_y_pred_list, axis=0)

    # Save results
    json_path = os.path.join(save_dir, f"{dataset_name}_global_results.json")
    with open(json_path, "w") as f:
        json.dump(global_log, f, indent=2)
    print(f"Saved global results to {json_path}")

    # Plotting
    module_names = list(clients[0].model.modules_to_integrate.keys())
    _plot_all_metrics(global_log, save_dir, dataset_name)
    _plot_training_loss(global_log, save_dir, dataset_name)
    _plot_module_replacement(global_log, save_dir, dataset_name, module_names)
    _plot_temporal_stability(final_y_true, final_y_pred, save_dir, dataset_name)
    _plot_true_vs_predicted(final_y_true, final_y_pred, save_dir, dataset_name)
    _plot_error_distribution(final_y_true, final_y_pred, save_dir, dataset_name)
    
    if vmd_k > 0 and original_signal_for_plot is not None:
        # Re-evaluate final model on first client's test set to get preds for frequency plot
        # The returned arrays will have shape [samples, horizon, nodes_in_client, features]
        _, _, _, _, _, y_true_client, y_pred_client = clients[0].test_metrics()
        
        # Select ONLY the first node for this client for a 1-to-1 comparison
        y_true_for_freq = y_true_client[:, :, 0, :].flatten() # Select first node and flatten
        y_pred_for_freq = y_pred_client[:, :, 0, :].flatten() # Select first node and flatten

        _plot_frequency_domain_analysis(
            y_true_for_freq, 
            vmd_reconstruction_for_plot[:len(y_true_for_freq)], 
            y_pred_for_freq, 
            save_dir, 
            dataset_name
        )
    # --- Final Tables ---
    final_metrics = {
        "Dataset": dataset_name,
        "MAE": global_log[-1]["global_test_mae"],
        "RMSE": global_log[-1]["global_test_rmse"],
        "MAPE (%)": global_log[-1]["global_test_mape"],
        "MASE": global_log[-1]["global_test_mase"],
        "R^2": global_log[-1]["global_test_r2"],
    }
    print("\n--- Performance Comparison ---")
    print(pd.DataFrame([final_metrics]).set_index("Dataset").to_markdown(floatfmt=".4f"))
    
    total_runtime = time.time() - start_time
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    runtime_and_memory = {
        "Configuration": dataset_name,
        "Total Runtime (s)": total_runtime,
        "Peak Memory (MB)": peak_mem / 1024 / 1024,
        "VMD (K)": vmd_k,
        "Attention": "Yes" if use_attention else "No",
        "Adaptive Adjacency": "No (Ablated)" if ablate_adja else "Yes",
    }
    print("\n--- Computational Metrics ---")
    print(pd.DataFrame([runtime_and_memory]).set_index("Configuration").to_markdown(floatfmt=".2f"))

    print("\n--- Notes on Generating Comparative Plots/Tables ---")
    print("1. Ablation Study: To generate the full 'Ablation study table', run this script multiple times with different flags:")
    print("   - Base model: --vmd_k 0 --use_attention 0 --ablate_adja 1")
    print("   - + VMD: --vmd_k 3 --use_attention 0 --ablate_adja 1")
    print("   - + Attention: --vmd_k 0 --use_attention 1 --ablate_adja 1")
    print("   - + Adaptive Adj: --vmd_k 0 --use_attention 0 --ablate_adja 0")
    print("   - Full model: --vmd_k 3 --use_attention 1 --ablate_adja 0")
    print("2. Complexity vs. Gain Plot: This plot is conceptual. After running the ablation studies, you can manually plot 'MAE' (gain) vs. 'Total Runtime (s)' or 'Peak Memory (MB)' (complexity) using the results from the tables above.")


# ================================================================
# CLI
# ================================================================
def main():
    ap = argparse.ArgumentParser(description="Federated Time Series Forecasting with LSTM-DSTGCRN")
    ap.add_argument("--trip_csv", required=True, help="Path to *tripdata*_full.csv (e.g., 'CHI-taxi/tripdata_full.csv')")
    ap.add_argument("--dataset", default="Dataset", help="Name for the dataset for titles and filenames (e.g., 'NYC-Bike')")
    ap.add_argument("--rounds", type=int, default=10, help="Number of federated rounds")
    ap.add_argument("--clients", type=int, default=3, help="Number of clients")
    ap.add_argument("--tin", type=int, default=12, help="Length of input sequence")
    ap.add_argument("--tout", type=int, default=3, help="Length of output horizon")
    ap.add_argument("--epochs_per_round", type=int, default=2, help="Local epochs per client per round")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--save_dir", default="./results", help="Directory to save checkpoints, logs, and plots")
    # --- Model Configuration ---
    ap.add_argument("--vmd_k", type=int, default=0, help="If >0, use VMD with K modes. Model predicts K IMFs and reconstructs for loss/metrics")
    ap.add_argument("--use_attention", type=int, default=0, help="Use 1 to enable multi-head attention, 0 to disable")
    ap.add_argument("--attn_heads", type=int, default=2, help="Number of attention heads")
    # --- Ablation ---
    ap.add_argument("--ablate_adja", type=int, default=0, help="Use 1 to replace adaptive adjacency with identity matrix for ablation study")
    # --- Other ---
    ap.add_argument("--save_vmd", type=int, default=1, help="Save VMD-preprocessed file once (client 0)")
    args = ap.parse_args()

    run_federated(
        trip_csv=args.trip_csv,
        dataset_name=args.dataset,
        num_clients=args.clients,
        num_rounds=args.rounds,
        input_len=args.tin,
        output_len=args.tout,
        epochs_per_round=args.epochs_per_round,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        use_attention=bool(args.use_attention),
        attn_heads=args.attn_heads,
        vmd_k=args.vmd_k,
        save_vmd=bool(args.save_vmd),
        ablate_adja=bool(args.ablate_adja),
    )

if __name__ == "__main__":
    main()