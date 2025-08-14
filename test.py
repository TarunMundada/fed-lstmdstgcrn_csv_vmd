"""
Minimal, modular PyTorch implementation of a Federated LSTM-DSTGCRN
with FedAvg and client-side validation-based selective integration (Algorithm 2).

This is a teaching/reference scaffold intended to be easy to adapt to
real datasets (NYC-bike/taxi, CHI-taxi, etc.). Replace the dataset
stubs with your loaders and metrics.

Key pieces:
- Adaptive/Dynamic graph constructor from node embeddings
- GraphConv (message passing via learned dynamic adjacency)
- GraphConvLSTMCell (injects graph conv into LSTM gates)
- LSTM-DSTGCRN model (stacked cells + projection)
- Federated Client/Server with:
  * Local training per round (E epochs)
  * FedAvg aggregation (weighted by sample count)
  * Algorithm 2: layer-wise validation-driven selective integration
- Round/epoch checkpointing and y_true/y_pred capture hooks

Author: ChatGPT
License: MIT
"""
from __future__ import annotations
import os
import copy
import math
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ================================================================
# Utils
# ================================================================

def set_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(x, device):
    if isinstance(x, (list, tuple)):
        return [to_device(xx, device) for xx in x]
    return x.to(device)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ================================================================
# Dynamic (Adaptive) Graph Builder (learned adjacency)
# ================================================================
class AdaptiveAdjacency(nn.Module):
    """
    Builds a dynamic adjacency matrix from learnable node embeddings.
    A = softmax( ReLU(E1 @ E2^T) )  # row-normalized
    This can change over time if you pass a gating signal (optional).
    """
    def __init__(self, num_nodes: int, emb_dim: int):
        super().__init__()
        self.E1 = nn.Parameter(torch.randn(num_nodes, emb_dim) * 0.1)
        self.E2 = nn.Parameter(torch.randn(num_nodes, emb_dim) * 0.1)

    def forward(self) -> torch.Tensor:
        logits = F.relu(self.E1 @ self.E2.t())  # [N,N]
        A = F.softmax(logits, dim=-1)
        return A


# ================================================================
# Graph Convolution via learned adjacency (simple message passing)
# ================================================================
class GraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, C]
        A: [N, N] (row-normalized)
        Returns: [B, N, out_channels]
        """
        # message passing: X' = A @ X  (per batch)
        x_mp = torch.einsum("ij,bjc->bic", A, x)
        return self.lin(x_mp)


# ================================================================
# GraphConvLSTM Cell
# ================================================================
class GraphConvLSTMCell(nn.Module):
    """
    An LSTM cell where the input->hidden transforms are graph-convolved.

    h_t, c_t = LSTM(GC(x_t), h_{t-1}, c_{t-1})
    """
    def __init__(self, num_nodes: int, in_dim: int, hidden_dim: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.gc_x = GraphConv(in_dim, 4 * hidden_dim)
        self.gc_h = GraphConv(hidden_dim, 4 * hidden_dim)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor, A: torch.Tensor):
        # x_t: [B,N,Cin], h_prev/c_prev: [B,N,H]
        gates = self.gc_x(x_t, A) + self.gc_h(h_prev, A)  # [B,N,4H]
        i, f, o, g = torch.chunk(gates, chunks=4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)
        return h_t, c_t


# ================================================================
# LSTM-DSTGCRN Model
# ================================================================
class LSTMDSTGCRN(nn.Module):
    """
    Stacked GraphConvLSTM with adaptive adjacency and linear projection.
    Input:  x of shape [B, T_in, N, C_in]
    Output: y of shape [B, T_out, N, C_out]
    """
    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        horizon: int = 1,
        num_layers: int = 1,
        emb_dim: int = 10,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon
        self.adaptiveA = AdaptiveAdjacency(num_nodes, emb_dim)

        cells = []
        for l in range(num_layers):
            in_dim = input_dim if l == 0 else hidden_dim
            cells.append(GraphConvLSTMCell(num_nodes, in_dim, hidden_dim))
        self.cells = nn.ModuleList(cells)

        self.proj = nn.Linear(hidden_dim, output_dim * horizon)

        # For Algorithm 2 selective integration, we expose layer groups
        self.modules_to_integrate = {
            "adaptiveA": self.adaptiveA,
            "cells": self.cells,
            "proj": self.proj,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,N,C]
        B, T, N, C = x.shape
        assert N == self.num_nodes
        A = self.adaptiveA()  # [N,N]

        # initialize hidden states
        hs = [x.new_zeros(B, N, cell.hidden_dim) for cell in self.cells]
        cs = [x.new_zeros(B, N, cell.hidden_dim) for cell in self.cells]

        # encode over time
        for t in range(T):
            xt = x[:, t]
            for l, cell in enumerate(self.cells):
                h, c = cell(xt, hs[l], cs[l], A)
                hs[l], cs[l] = h, c
                xt = h  # feed to next layer

        # use final h of top layer for projection -> horizon
        out = self.proj(hs[-1])  # [B,N,output_dim*horizon]
        out = out.view(B, N, self.horizon, -1).permute(0, 2, 1, 3)  # [B,T_out,N,C_out]
        return out


# ================================================================
# Real dataset loader (matches original project's preprocessing)
# ================================================================
class TripWeatherDataset(Dataset):
    """
    Loader adapted from the original `load_and_transform_data` you shared.

    - Does NOT merge DataFrames; instead it reads trips and weather separately,
      parses timestamps (drops tzinfo), and uses positional slicing of weather
      columns (temperature, precipitation) as in the original code.
    - Produces windows with features:
        [trip_count, temperature, precipitation, hour_norm, day_in_week_norm, weekend_flag]
      Shape per sample: x: [T_in, N, F], y: [T_out, N, 1]

    Usage: TripWeatherDataset(trip_csv, input_len=12, output_len=3, stride=1)
    """

    def __init__(
        self,
        trip_csv: str,
        input_len: int = 12,
        output_len: int = 3,
        stride: int = 1,
        fillna: float = 0.0,
    ):
        import pandas as pd
        import numpy as np

        self.T_in = input_len
        self.T_out = output_len
        self.stride = stride

        trips = pd.read_csv("CHI-taxi/tripdata_full.csv")
        weathers = pd.read_csv("CHI-taxi/weatherdata_full.csv")

        # Parse timestamps and drop tzinfo to avoid timezone dtype mismatch
        trips['timestamp'] = pd.to_datetime(trips['timestamp']).dt.tz_localize(None)
        weathers['timestamp'] = pd.to_datetime(weathers['timestamp']).dt.tz_localize(None)

        # Set index for easy time-based features (we won't merge on timestamps)
        trips = trips.set_index('timestamp')
        weathers = weathers.set_index('timestamp')

        # Convert to numpy arrays; trips_np shape: [T, N]
        trips_np = trips.to_numpy()
        weathers_np = weathers.to_numpy()  # expected shape [T, 2*N] (temp, precip alternating)

        # Create time features like the original loader
        # Weekend one-hot -> take the second column (original code used encoder and [:,1])
        from sklearn.preprocessing import OneHotEncoder
        encoder = OneHotEncoder(sparse_output=False)
        weekends = trips.index.dayofweek.isin([5, 6])
        weekend_1hot = encoder.fit_transform(weekends.reshape(-1, 1))[:, 1].reshape(-1, 1)
        weekend_1hot = np.repeat(weekend_1hot[:, np.newaxis, :], trips.shape[1], axis=1)  # [T, N, 1]

        day_in_week = trips.index.dayofweek.to_numpy()
        day_in_week_normalized = day_in_week / 7.0
        day_in_week_reshaped = np.repeat(day_in_week_normalized[:, np.newaxis], trips.shape[1], axis=1)

        hours = trips.index.hour.to_numpy()
        hour_normalized = hours / 24.0
        hour_reshaped = np.repeat(hour_normalized[:, np.newaxis], trips.shape[1], axis=1)

        # Slice weather: temperature = [:, ::2], precipitation = [:, 1::2]
        temperature = weathers_np[:, ::2]
        precipitation = weathers_np[:, 1::2]

        # Build feature tensor exactly like the original function did
        data = np.concatenate(
            (
                trips_np[:, :, np.newaxis],            # trips
                temperature[:, :, np.newaxis],          # temp
                precipitation[:, :, np.newaxis],        # precip
                hour_reshaped[:, :, np.newaxis],        # hour
                day_in_week_reshaped[:, :, np.newaxis], # day in week
                weekend_1hot,                           # weekend flag
            ),
            axis=2,
        )  # shape [T, N, F]

        # Cast to torch and keep for windows
        self.data = torch.tensor(data, dtype=torch.float32)

        # Targets are trip counts (first channel) for the forecast horizon
        self.targets = self.data[:, :, 0:1]  # [T, N, 1]

        # Build sliding windows
        T = self.data.shape[0]
        self.windows = []
        for start in range(0, T - (self.T_in + self.T_out) + 1, self.stride):
            x = self.data[start : start + self.T_in]  # [T_in, N, F]
            y = self.targets[start + self.T_in : start + self.T_in + self.T_out]  # [T_out, N, 1]
            self.windows.append((x, y))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]

# Collate remains the same (for [B,T,N,C])
def collate_batch(batch):
    xs, ys = zip(*batch)
    x = torch.stack(xs, dim=0)
    y = torch.stack(ys, dim=0)
    return x, y
    xs, ys = zip(*batch)
    x = torch.stack(xs, dim=0)
    y = torch.stack(ys, dim=0)
    return x, y


# ================================================================
# Training / Evaluation helpers
# ================================================================

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = to_device(x, device), to_device(y, device)
        optimizer.zero_grad()
        y_hat = model(x)
        loss = F.l1_loss(y_hat, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    total = 0.0
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = to_device(x, device), to_device(y, device)
            y_hat = model(x)
            loss = F.l1_loss(y_hat, y)
            total += loss.item() * x.size(0)
            y_true_all.append(y.cpu())
            y_pred_all.append(y_hat.cpu())
    y_true = torch.cat(y_true_all, dim=0)
    y_pred = torch.cat(y_pred_all, dim=0)
    return total / len(loader.dataset), y_true, y_pred


# ================================================================
# Algorithm 2: Client-side validation-based selective integration
# ================================================================

def selective_integration(
    local_model: nn.Module,
    global_model: nn.Module,
    val_loader: DataLoader,
    device,
    candidate_groups: Optional[List[str]] = None,
) -> nn.Module:
    """
    For each layer (or group) S in Z, replace S in local with global[S],
    evaluate on validation set, pick argmin loss, keep that replacement.
    """
    local = local_model
    best_loss, best_group = math.inf, None

    # Named groups exposed in model.modules_to_integrate
    groups = list(local.modules_to_integrate.keys()) if candidate_groups is None else candidate_groups

    base_sd = local.state_dict()
    for g in groups:
        candidate = copy.deepcopy(local)
        # Copy the corresponding group params from global -> candidate
        _copy_group(candidate, global_model, g)
        val_loss, _, _ = evaluate(candidate, val_loader, device)
        if val_loss < best_loss:
            best_loss, best_group = val_loss, g

    # Apply best group replacement to the true local model
    if best_group is not None:
        _copy_group(local, global_model, best_group)

    return local


def _copy_group(dst: nn.Module, src: nn.Module, group_name: str):
    dst_group = dst.modules_to_integrate[group_name]
    src_group = src.modules_to_integrate[group_name]
    dst_group.load_state_dict(src_group.state_dict())


# ================================================================
# Federated classes
# ================================================================
@dataclass
class ClientConfig:
    id: int
    epochs: int
    batch_size: int
    lr: float
    n_samples: int


class FLClient:
    def __init__(self, cid: int, model_fn, train_ds: Dataset, val_ds: Dataset, test_ds: Dataset, cfg: ClientConfig, device):
        self.cid = cid
        self.device = device
        self.cfg = cfg
        self.model = model_fn().to(device)
        self.train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_batch)
        self.val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_batch)
        self.test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_batch)

    def set_parameters(self, global_model: nn.Module):
        self.model.load_state_dict(global_model.state_dict())

    def integrate_with_validation(self, global_model: nn.Module):
        self.model = selective_integration(self.model, global_model, self.val_loader, self.device)

    def train_local(self) -> Dict:
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        hist = {"train_loss": [], "val_loss": []}
        for e in range(self.cfg.epochs):
            tr = train_one_epoch(self.model, self.train_loader, opt, self.device)
            vl, _, _ = evaluate(self.model, self.val_loader, self.device)
            hist["train_loss"].append(tr)
            hist["val_loss"].append(vl)
        return hist

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        return copy.deepcopy(self.model.state_dict())

    def evaluate_test(self) -> Tuple[float, torch.Tensor, torch.Tensor]:
        return evaluate(self.model, self.test_loader, self.device)


class FLServer:
    def __init__(self, model_fn, clients: List[FLClient], save_dir: str = "./ckpts"):
        self.global_model = model_fn()
        self.clients = clients
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def distribute(self):
        for c in self.clients:
            c.set_parameters(self.global_model)

    def aggregate_fedavg(self, client_state_dicts: List[Tuple[Dict, int]]):
        # Weighted average by n_samples
        total = sum(n for _, n in client_state_dicts)
        new_sd = copy.deepcopy(client_state_dicts[0][0])
        for k in new_sd.keys():
            new_sd[k] = sum(sd[k] * (n/total) for sd, n in client_state_dicts)
        self.global_model.load_state_dict(new_sd)

    def checkpoint_round(self, round_idx: int):
        path = os.path.join(self.save_dir, f"global_round_{round_idx:03d}.pt")
        torch.save(self.global_model.state_dict(), path)
        return path


# ================================================================
# Main Federated loop (Algorithm 1)
# ================================================================

def run_federated(
    num_rounds: int = 5,
    num_clients: int = 3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_dir: str = "./ckpts",
    trip_csv: str = "CHI-taxi/tripdata_full.csv",
    weather_csv: str = "CHI-taxi/weatherdata_full.csv",
    input_len: int = 12,
    output_len: int = 3,
):
    set_seed(42)

    # ---------- Define node partitions per client
    # Infer node ids from CSV headers (assumes cols like '1.0' or '1')
    import pandas as pd
    tmp = pd.read_csv(trip_csv, nrows=1)
    node_cols = [c for c in tmp.columns if c != 'timestamp']
    # normalize to int ids
    def to_id(c):
        try:
            return int(float(c))
        except Exception:
            return None
    node_ids = [to_id(c) for c in node_cols]
    node_ids = [n for n in node_ids if n is not None]
    node_ids = sorted(set(node_ids))

    if len(node_ids) == 0:
        raise ValueError("No node columns found in tripdata CSV.")

    # simple even split of nodes across clients
    chunks = [node_ids[i::num_clients] for i in range(num_clients)]

    NODES = len(node_ids)
    TIN, TOUT = input_len, output_len

    def model_fn():
        return LSTMDSTGCRN(num_nodes=NODES, input_dim=6, hidden_dim=64, output_dim=1, horizon=TOUT, num_layers=1, emb_dim=16)

    # Build clients: each sees *all timestamps* but only its subset of nodes
    clients: List[FLClient] = []
    for cid, nodes in enumerate(chunks):
        ds = TripWeatherDataset(trip_csv, input_len=TIN, output_len=TOUT, stride=1)
        n_total = len(ds)
        n_tr = int(0.7 * n_total)
        n_va = int(0.15 * n_total)
        n_te = n_total - n_tr - n_va
        tr, va, te = torch.utils.data.random_split(ds, [n_tr, n_va, n_te], generator=torch.Generator().manual_seed(42+cid))
        cfg = ClientConfig(id=cid, epochs=20, batch_size=32, lr=1e-3, n_samples=n_tr)
        clients.append(FLClient(cid, model_fn, tr, va, te, cfg, device))

    server = FLServer(model_fn, clients, save_dir)

    # ---------- Federated rounds
    metrics_per_round = []
    for r in range(num_rounds):
        print(f"================= FL ROUND {r+1}/{num_rounds} =================")
        server.distribute()

        client_states = []
        for c in clients:
            c.integrate_with_validation(server.global_model)
            hist = c.train_local()
            print(f"Client {c.cid}: train_loss={hist['train_loss'][-1]:.4f}, val_loss={hist['val_loss'][-1]:.4f}")
            client_states.append((c.get_parameters(), c.cfg.n_samples))

        server.aggregate_fedavg(client_states)
        ckpt = server.checkpoint_round(r)
        print(f"Saved global checkpoint: {ckpt}")

        server.distribute()
        round_report = {"round": r}
        for c in clients:
            test_loss, y_true, y_pred = c.evaluate_test()
            round_report[f"client_{c.cid}_test_mae"] = float(test_loss)
            out_dir = os.path.join(save_dir, f"round_{r:03d}")
            os.makedirs(out_dir, exist_ok=True)
            torch.save({"y_true": y_true, "y_pred": y_pred}, os.path.join(out_dir, f"client_{c.cid}_yp.pt"))
        metrics_per_round.append(round_report)

    with open(os.path.join(save_dir, "metrics_per_round.json"), "w") as f:
        json.dump(metrics_per_round, f, indent=2)


# ================================================================
# Entry point for quick sanity check
# ================================================================
if __name__ == "__main__":
    run_federated(num_rounds=10, num_clients=3, device="cuda" if torch.cuda.is_available() else "cpu")
    # run_federated(num_rounds=3, num_clients=3, device="cuda" if torch.cuda.is_available() else "cpu")
