"""
Federated LSTM-DSTGCRN — fixed per-client node-subset handling.
Extended baseline to (1) log both MAE and RMSE and (2) optionally apply
multi-head temporal self-attention before the GraphConvLSTM.

Run example:
python federated_lstm_dstgcrn_baseline.py --trip_csv CHI-taxi/tripdata_full.csv \
  --dataset CHI-Taxi --rounds 10 --clients 3 --tin 12 --tout 3 \
  --epochs_per_round 2 --batch_size 16 --use_attention 1
"""
from __future__ import annotations
import os, copy, math, json, argparse, random
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
    def __init__(self, num_nodes: int, emb_dim: int):
        super().__init__()
        self.E1 = nn.Parameter(torch.randn(num_nodes, emb_dim) * 0.1)
        self.E2 = nn.Parameter(torch.randn(num_nodes, emb_dim) * 0.1)

    def forward(self):
        logits = F.relu(self.E1 @ self.E2.t())            # [N,N]
        return F.softmax(logits, dim=-1)                  # row-normalized

class GraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # x: [B,N,C], A: [N,N]
        x_mp = torch.einsum("ij,bjc->bic", A, x)          # message passing
        return self.lin(x_mp)

# ================================================================
# Optional: Multi-head temporal self-attention (per node)
# ================================================================
class TemporalMHA(nn.Module):
    """Applies MHA along time for each node independently.
    Input  x: [B,T,N,C] -> output [B,T,N,C]
    Uses nn.MultiheadAttention with batch_first=True.
    """
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                         dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, C = x.shape
        x_ = x.permute(0, 2, 1, 3).reshape(B * N, T, C)  # [B*N,T,C]
        out, _ = self.mha(x_, x_, x_, need_weights=False)
        out = self.ln(out + x_)
        out = out.reshape(B, N, T, C).permute(0, 2, 1, 3)  # [B,T,N,C]
        return out

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
        attn_heads: int = 4,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon
        self.adaptiveA = AdaptiveAdjacency(num_nodes, emb_dim)
        self.use_attention = use_attention
        if use_attention:
            self.attn = TemporalMHA(dim=input_dim, num_heads=attn_heads)

        cells = []
        for l in range(num_layers):
            in_dim = input_dim if l == 0 else hidden_dim
            cells.append(GraphConvLSTMCell(num_nodes, in_dim, hidden_dim))
        self.cells = nn.ModuleList(cells)

        self.proj = nn.Linear(hidden_dim, output_dim * horizon)

        # expose groups for Algorithm 2
        self.modules_to_integrate = {
            "adaptiveA": self.adaptiveA,
            "cells": self.cells,
            "proj": self.proj,
        }
        if use_attention:
            self.modules_to_integrate["attn"] = self.attn

    def forward(self, x):
        # x: [B,T,N,C]
        B, T, N, C = x.shape
        assert N == self.num_nodes, f"Input N={N} but model expects {self.num_nodes}"
        if self.use_attention:
            x = self.attn(x)  # temporal self-attention per node

        A = self.adaptiveA()  # [N,N]

        hs = [x.new_zeros(B, N, c.hidden_dim) for c in self.cells]
        cs = [x.new_zeros(B, N, c.hidden_dim) for c in self.cells]

        for t in range(T):
            xt = x[:, t]
            for i, cell in enumerate(self.cells):
                h, c = cell(xt, hs[i], cs[i], A)
                hs[i], cs[i] = h, c
                xt = h

        out = self.proj(hs[-1])                     # [B,N, C_out*horizon]
        out = out.view(B, N, self.horizon, -1).permute(0, 2, 1, 3)  # [B,T_out,N,C_out]
        return out

# ================================================================
# Dataset (matches original loader; optional node subset)
# ================================================================
class TripWeatherDataset(Dataset):
    """
    Builds features like the original code:
      [trips, temperature, precipitation, hour_norm, day_in_week_norm, weekend_flag]
    If node_subset is provided (list of column indices), the dataset will return windows with N_sub nodes.
    """
    def __init__(
        self,
        trip_csv: str,
        input_len: int = 12,
        output_len: int = 3,
        stride: int = 1,
        node_subset: Optional[List[int]] = None,  # indices 0..N-1
    ):
        self.T_in = input_len
        self.T_out = output_len
        self.stride = stride
        self.node_subset = None if node_subset is None else list(node_subset)

        trips = pd.read_csv(trip_csv)
        weather_csv = trip_csv.replace("tripdata", "weatherdata")
        weathers = pd.read_csv(weather_csv)

        # parse timestamps & drop tz
        trips["timestamp"] = pd.to_datetime(trips["timestamp"]).dt.tz_localize(None)
        weathers["timestamp"] = pd.to_datetime(weathers["timestamp"]).dt.tz_localize(None)

        trips = trips.set_index("timestamp")
        weathers = weathers.set_index("timestamp")

        trips_np = trips.to_numpy()          # [T, N]
        weathers_np = weathers.to_numpy()    # [T, 2N] -> (temp, precip) alternating

        # time features
        weekends = trips.index.dayofweek.isin([5, 6])
        enc = OneHotEncoder(sparse_output=False)
        weekend_1hot = enc.fit_transform(weekends.reshape(-1, 1))[:, 1].reshape(-1, 1)
        weekend_1hot = np.repeat(weekend_1hot[:, np.newaxis, :], trips_np.shape[1], axis=1)  # [T,N,1]

        day_norm = (trips.index.dayofweek.to_numpy() / 7.0)
        day_norm = np.repeat(day_norm[:, None], trips_np.shape[1], axis=1)

        hour_norm = (trips.index.hour.to_numpy() / 24.0)
        hour_norm = np.repeat(hour_norm[:, None], trips_np.shape[1], axis=1)

        temperature = weathers_np[:, ::2]
        precipitation = weathers_np[:, 1::2]

        data = np.concatenate(
            (
                trips_np[:, :, None],
                temperature[:, :, None],
                precipitation[:, :, None],
                hour_norm[:, :, None],
                day_norm[:, :, None],
                weekend_1hot,
            ),
            axis=2,
        )  # [T,N,F]

        # If node_subset provided, keep only those columns for returned windows
        if self.node_subset is not None:
            data_sub = data[:, self.node_subset, :]   # [T, N_sub, F]
        else:
            data_sub = data  # full

        self.data = torch.tensor(data_sub, dtype=torch.float32)
        self.targets = self.data[:, :, 0:1]  # trips channel

        # windows
        T = self.data.shape[0]
        self.windows = []
        for s in range(0, T - (self.T_in + self.T_out) + 1, self.stride):
            x = self.data[s : s + self.T_in]                        # [T_in, N_sub, F]
            y = self.targets[s + self.T_in : s + self.T_in + self.T_out]  # [T_out, N_sub, 1]
            self.windows.append((x, y))

    def __len__(self): return len(self.windows)
    def __getitem__(self, idx): return self.windows[idx]

def collate_batch(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0)  # x:[B,T,N_sub,F], y:[B,T_out,N_sub,1]

# ================================================================
# Padding helper (pad batch from N_sub -> N_full according to indices)
# ================================================================
def pad_batch_to_full(x_sub: torch.Tensor, subset_idx: List[int], N_full: int):
    """
    x_sub: [B,T,N_sub,C], subset_idx: list length N_sub with positions in 0..N_full-1
    returns:
      x_full: [B,T,N_full,C]
      mask:   [B,T,N_full,1] (1 for real node positions, 0 otherwise)
    """
    B, T, N_sub, C = x_sub.shape
    x_full = x_sub.new_zeros((B, T, N_full, C))
    mask = x_sub.new_zeros((B, T, N_full, 1))
    for i, pos in enumerate(subset_idx):
        x_full[:, :, pos, :] = x_sub[:, :, i, :]
        mask[:, :, pos, 0] = 1.0
    return x_full, mask

# ================================================================
# Train / Eval helpers (now compute MAE & RMSE)
# ================================================================
def _masked_mae(y_hat, y, mask=None):
    if mask is None:
        return (y_hat - y).abs().mean()
    else:
        abs_diff = (y_hat - y).abs() * mask
        return abs_diff.sum() / mask.sum().clamp_min(1.0)


def _masked_rmse(y_hat, y, mask=None):
    if mask is None:
        return torch.sqrt(((y_hat - y) ** 2).mean())
    else:
        sq = ((y_hat - y) ** 2) * mask
        return torch.sqrt(sq.sum() / mask.sum().clamp_min(1.0))


def train_one_epoch(model, loader, optimizer, device, subset_idx=None, N_full=None):
    model.train()
    total_loss = 0.0
    total_count = 0

    for x, y in loader:
        if subset_idx is not None:
            x_full, mask = pad_batch_to_full(x, subset_idx, N_full)
            y_full, _ = pad_batch_to_full(y, subset_idx, N_full)
            if mask.shape[1] != y_full.shape[1]:
                mask = mask[:, :y_full.shape[1], :, :]
            x, y, mask = to_device(x_full, device), to_device(y_full, device), to_device(mask, device)
        else:
            mask = None
            x, y = to_device(x, device), to_device(y, device)

        optimizer.zero_grad()
        y_hat = model(x)
        loss = _masked_mae(y_hat, y, mask)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_count += x.size(0)

    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate(model, loader, device, subset_idx=None, N_full=None):
    model.eval()
    total_mae = 0.0
    total_rmse = 0.0
    total_batches = 0

    for x, y in loader:
        if subset_idx is not None:
            x_full, mask = pad_batch_to_full(x, subset_idx, N_full)
            y_full, _ = pad_batch_to_full(y, subset_idx, N_full)
            if mask.shape[1] != y_full.shape[1]:
                mask = mask[:, :y_full.shape[1], :, :]
            x, y, mask = to_device(x_full, device), to_device(y_full, device), to_device(mask, device)
        else:
            mask = None
            x, y = to_device(x, device), to_device(y, device)

        y_hat = model(x)
        mae = _masked_mae(y_hat, y, mask)
        rmse = _masked_rmse(y_hat, y, mask)
        total_mae += mae.item()
        total_rmse += rmse.item()
        total_batches += 1

    if total_batches == 0:
        return {"mae": float("nan"), "rmse": float("nan")}
    return {"mae": total_mae / total_batches, "rmse": total_rmse / total_batches}


# ================================================================
# Algorithm 2: selective integration (evaluates candidate models on padded validation batches)
# ================================================================
def _copy_group(dst: nn.Module, src: nn.Module, group_name: str):
    dst_group = dst.modules_to_integrate[group_name]
    src_group = src.modules_to_integrate[group_name]
    dst_group.load_state_dict(src_group.state_dict())


def selective_integration(local_model, global_model, val_loader, device, subset_idx: Optional[List[int]] = None, N_full: Optional[int] = None, groups=None):
    if groups is None:
        groups = list(local_model.modules_to_integrate.keys())

    best_loss = math.inf
    best_group = None
    base = copy.deepcopy(local_model)

    for g in groups:
        cand = copy.deepcopy(base)
        _copy_group(cand, global_model, g)
        metrics = evaluate(cand, val_loader, device, subset_idx=subset_idx, N_full=N_full)
        loss = metrics["mae"]
        if loss < best_loss:
            best_loss, best_group = loss, g

    if best_group is not None:
        _copy_group(local_model, global_model, best_group)
    return local_model

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

    def set_params_from(self, global_model):
        self.model.load_state_dict(global_model.state_dict())

    def integrate(self, global_model):
        self.model = selective_integration(self.model, global_model, self.val_loader, self.device, subset_idx=self.subset_idx, N_full=self.N_full)

    def train_local(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        for _ in range(self.cfg.epochs):
            train_one_epoch(self.model, self.train_loader, opt, self.device, subset_idx=self.subset_idx, N_full=self.N_full)

    def state_dict(self):
        return copy.deepcopy(self.model.state_dict())

    def n_train(self): return len(self.train_loader.dataset)
    def n_test(self):  return len(self.test_loader.dataset)

    @torch.no_grad()
    def test_metrics(self):
        return evaluate(self.model, self.test_loader, self.device, subset_idx=self.subset_idx, N_full=self.N_full)

    @torch.no_grad()
    def train_metrics(self):
        return evaluate(self.model, self.train_loader, self.device, subset_idx=self.subset_idx, N_full=self.N_full)

class FLServer:
    def __init__(self, model_fn, N_full: int):
        self.global_model = model_fn()
        self.N_full = N_full

    def distribute(self, clients: List[FLClient]):
        for c in clients:
            c.set_params_from(self.global_model)

    def aggregate_fedavg(self, states: List[Tuple[Dict, int]]):
        total = sum(n for _, n in states)
        new_sd = copy.deepcopy(states[0][0])
        for k in new_sd.keys():
            new_sd[k] = sum(sd[k] * (n/total) for sd, n in states)
        self.global_model.load_state_dict(new_sd)

# ================================================================
# Federated loop with global metrics + plotting
# ================================================================
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
    use_attention: int = 0,
    attn_heads: int = 3,
):
    os.makedirs(save_dir, exist_ok=True)
    set_seed(42)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # infer N (nodes) and build node index list [0..N-1]
    tmp = pd.read_csv(trip_csv, nrows=1)
    node_cols = [c for c in tmp.columns if c != "timestamp"]
    N_full = len(node_cols)
    node_indices = list(range(N_full))

    # round-robin split nodes among clients
    chunks = [node_indices[i::num_clients] for i in range(num_clients)]

    def model_fn():
        return LSTMDSTGCRN(num_nodes=N_full, input_dim=6, hidden_dim=64,
                           output_dim=1, horizon=output_len, num_layers=1, emb_dim=16,
                           use_attention=bool(use_attention), attn_heads=attn_heads)

    # build clients
    clients: List[FLClient] = []
    for cid, subset in enumerate(chunks):
        full_ds = TripWeatherDataset(trip_csv, input_len, output_len, stride=1, node_subset=subset)
        n_total = len(full_ds)
        if n_total == 0:
            raise ValueError(f"Client {cid} has no samples. Check node splitting and dataset shape.")
        n_tr = int(0.70 * n_total)
        n_va = int(0.15 * n_total)
        n_te = n_total - n_tr - n_va
        tr, va, te = torch.utils.data.random_split(
            full_ds, [n_tr, n_va, n_te], generator=torch.Generator().manual_seed(123 + cid)
        )
        cfg = ClientConfig(id=cid, epochs=epochs_per_round, batch_size=batch_size, lr=1e-3)
        clients.append(FLClient(cid, model_fn, tr, va, te, cfg, device, subset_idx=subset, N_full=N_full))

    server = FLServer(model_fn, N_full=N_full)

    # logs for paper-style
    global_log = []     # [{round, global_mae, global_rmse, global_train_mae, global_train_rmse}, ...]

    for r in range(num_rounds):
        print(f"\n===== Round {r+1}/{num_rounds} | {dataset_name} =====")

        # send current global model
        server.distribute(clients)

        # client-side selective integration + local training
        for c in clients:
            c.integrate(server.global_model)
            c.train_local()

        # Weighted global training metrics (evaluate on each client's TRAIN set)
        train_metrics = [(c.train_metrics(), c.n_train()) for c in clients]
        total_train = sum(n for _, n in train_metrics)
        global_train_mae = sum(m["mae"] * n for m, n in train_metrics) / total_train
        global_train_rmse = sum(m["rmse"] * n for m, n in train_metrics) / total_train

        # Evaluate test metrics per client
        test_results = [(c.test_metrics(), c.n_test()) for c in clients]
        total_test = sum(n for _, n in test_results)
        global_mae = sum(m["mae"] * n for m, n in test_results) / total_test
        global_rmse = sum(m["rmse"] * n for m, n in test_results) / total_test

        print(f"Global MAE: {global_mae:.4f} | Global RMSE: {global_rmse:.4f} | Train MAE: {global_train_mae:.4f}")
        global_log.append({
            "round": r,
            "global_mae": float(global_mae),
            "global_rmse": float(global_rmse),
            "global_train_mae": float(global_train_mae),
            "global_train_rmse": float(global_train_rmse),
        })

        # FedAvg aggregation (weighted by train samples)
        states = [(c.state_dict(), c.n_train()) for c in clients]
        server.aggregate_fedavg(states)

    # Save results for this dataset
    json_path = os.path.join(save_dir, f"{dataset_name}_global_results.json")
    with open(json_path, "w") as f:
        json.dump(global_log, f, indent=2)
    print(f"Saved global results to {json_path}")

    # Plot and save the curves for this dataset
    rounds = [r["round"] for r in global_log]
    maes = [r["global_mae"] for r in global_log]
    rmses = [r["global_rmse"] for r in global_log]

    plt.figure()
    plt.plot(rounds, maes, label=f"{dataset_name} MAE")
    plt.plot(rounds, rmses, label=f"{dataset_name} RMSE")
    plt.xlabel("Round")
    plt.ylabel("Metric")
    plt.title(f"Global Metrics ({dataset_name})")
    plt.legend()
    png_path = os.path.join(save_dir, f"{dataset_name}_metrics.png")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()
    print(f"Saved metrics curves to {png_path}")

# ================================================================
# CLI
# ================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trip_csv", required=True, help="Path to *tripdata*_full.csv (e.g., 'CHI-taxi/tripdata_full.csv')")
    ap.add_argument("--dataset", default="CHI-Taxi")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--clients", type=int, default=3)
    ap.add_argument("--tin", type=int, default=12)
    ap.add_argument("--tout", type=int, default=3)
    ap.add_argument("--epochs_per_round", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--use_attention", type=int, default=0, help="1 to enable temporal MHA pre-encoder")
    ap.add_argument("--attn_heads", type=int, default=4)
    ap.add_argument("--save_dir", default="./ckpts")
    args = ap.parse_args()

    run_federated(
        trip_csv=args.trip_csv,
        dataset_name=args.dataset,
        num_clients=args.clients,
        num_rounds=args.rounds,
        input_len=args.tin,
        output_len=args.tout,
        epochs_per_round=args.epochs_per_round,
        batch_size=args.batch_size,
        use_attention=args.use_attention,
        attn_heads=args.attn_heads,
        save_dir=args.save_dir,
    )

if __name__ == "__main__":
    main()
