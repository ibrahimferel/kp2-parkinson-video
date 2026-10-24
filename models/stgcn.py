from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from fogbench import data as D
from models.base import FoGModel

FS = 32
POSE_DIR = Path(os.environ.get("FOG_POSE_DIR", "pose_grid32"))
SKEMA = os.environ.get("FOG_STGCN_SKEMA", "pinggul")
EPOCHS = int(os.environ.get("FOG_STGCN_EPOCHS", "30"))
WIN, STRIDE, BATCH, LR, WD = 128, 32, 32, 5e-4, 1e-4
MIN_VIS = 0.5
DROPOUT = 0.25
V = 17
EDGES = [(0, 1), (0, 2), (1, 2), (1, 3), (3, 5), (2, 4), (4, 6), (1, 7), (2, 8), (7, 8), (7, 9), (9, 11),
         (8, 10), (10, 12), (11, 13), (11, 15), (13, 15), (12, 14), (12, 16), (14, 16)]
HIP = (7, 8)
_CACHE: dict = {}


def _muat(subject, session):
    key = (subject, session)
    if key not in _CACHE:
        d = np.load(POSE_DIR / f"PDFE{subject:02d}_{session + 1}.npz")
        _CACHE[key] = (d["xy"].astype(np.float32), d["vis"].astype(np.float32))
    return _CACHE[key]


def _selaraskan(a, n):
    if len(a) >= n:
        return a[:n]
    pad = np.full((n - len(a),) + a.shape[1:], np.nan, np.float32)
    return np.concatenate([a, pad])


def _skala_tubuh(xy):
    """Panjang torso (bahu-tengah ke pinggul-tengah), median per rekaman; cadangan: pinggul ke pergelangan kaki."""
    with np.errstate(all="ignore"):
        sh, hp, an = xy[:, 1:3].mean(1), xy[:, 7:9].mean(1), xy[:, 11:13].mean(1)
        s = np.nanmedian(np.linalg.norm(sh - hp, axis=1))
        if not np.isfinite(s) or s < 1:
            s = np.nanmedian(np.linalg.norm(hp - an, axis=1))
    return float(s) if np.isfinite(s) and s >= 1 else 1.0


def _praproses(xy, vis, skema):
    """xy (T,17,2) piksel -> (koordinat, valid (T,17))."""
    valid = np.isfinite(xy).all(-1) & (np.nan_to_num(vis) >= MIN_VIS)
    X = xy.copy()
    if skema in ("pinggul", "skala"):
        hip = X[:, HIP, :].mean(1)
        X = X - hip[:, None, :]
        valid &= np.isfinite(hip).all(-1)[:, None]
        if skema == "skala":
            X = X / _skala_tubuh(xy)
    return X, valid


def _ke_tensor(X, valid, mu, sd):
    Z = (X - mu) / sd
    Z = np.where(valid[..., None], Z, 0.0).astype(np.float32)
    return np.concatenate([Z, valid[..., None].astype(np.float32)], axis=-1)      # (T,17,3)


def bangun_A():
    """A: (3,V,V) partisi 'spatial configuration': akar, sentripetal, sentrifugal (jarak ke pinggul)."""
    adj = np.zeros((V, V))
    for i, j in EDGES:
        adj[i, j] = adj[j, i] = 1
    hop = np.full(V, np.inf)
    hop[list(HIP)] = 0
    for _ in range(V):
        for i, j in EDGES:
            hop[i], hop[j] = min(hop[i], hop[j] + 1), min(hop[j], hop[i] + 1)
    A = np.zeros((3, V, V), np.float32)
    A[0] = np.eye(V)
    for i in range(V):
        for j in range(V):
            if adj[i, j]:
                A[1 if hop[j] < hop[i] else 2 if hop[j] > hop[i] else 0][j, i] = 1
    deg = (A.sum(0)).sum(0) + 1e-6
    return torch.tensor(A / deg[None, None, :], dtype=torch.float32)


class BlokSTGCN(nn.Module):
    def __init__(self, cin, cout, A, dil=1, k_t=9, drop=DROPOUT):
        super().__init__()
        self.register_buffer("A", A)
        self.M = nn.Parameter(torch.ones_like(A))
        self.gc = nn.Conv2d(cin, cout * A.shape[0], 1)
        self.bn1 = nn.BatchNorm2d(cout)
        pad = (k_t - 1) // 2 * dil
        self.tc = nn.Sequential(nn.Conv2d(cout, cout, (k_t, 1), padding=(pad, 0), dilation=(dil, 1)),
                                nn.BatchNorm2d(cout), nn.Dropout(drop))
        self.res = nn.Identity() if cin == cout else nn.Sequential(nn.Conv2d(cin, cout, 1), nn.BatchNorm2d(cout))
        self.relu = nn.ReLU()
        self.K = A.shape[0]

    def forward(self, x):                                             # x: (N,C,T,V)
        n, _, t, v = x.shape
        y = self.gc(x).view(n, self.K, -1, t, v)
        y = torch.einsum("nkctv,kvw->nctw", y, self.A * self.M)
        y = self.relu(self.bn1(y))
        return self.relu(self.tc(y) + self.res(x))


class STGCN(nn.Module):
    """4 blok, dilasi temporal (1,1,2,4) -> medan reseptif ~65 sampel (~2 s); keluaran logit per sampel."""
    def __init__(self, cin=3, ch=(32, 32, 64, 64), dil=(1, 1, 2, 4)):
        super().__init__()
        A = bangun_A()
        self.bn0 = nn.BatchNorm1d(cin * V)
        blok, c = [], cin
        for co, d in zip(ch, dil):
            blok.append(BlokSTGCN(c, co, A, d)); c = co
        self.blok = nn.Sequential(*blok)
        self.head = nn.Conv1d(c, 1, 1)

    def forward(self, x):                                             # x: (N,T,V,C)
        n, t, v, c = x.shape
        x = x.permute(0, 3, 2, 1).reshape(n, c * v, t)
        x = self.bn0(x).view(n, c, v, t).permute(0, 1, 3, 2)          # (N,C,T,V)
        x = self.blok(x).mean(-1)                                     # pool sendi -> (N,C,T)
        return self.head(x).squeeze(1)                                # (N,T)


class STGCNFoG(FoGModel):
    def __init__(self, seed=0, skema=SKEMA, epochs=EPOCHS):
        self.seed, self.skema, self.epochs = seed, skema, epochs
        self.name = f"stgcn-{skema}"
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.net, self.mu, self.sd = None, None, None
        self.riwayat = []

    def _rekaman(self, r):
        n32 = len(D.to_eval_rate(r.y, r.fs, FS, is_label=True))
        xy, vis = _muat(r.subject, r.session)
        return _selaraskan(xy, n32), _selaraskan(vis, n32), n32

    def fit(self, recordings):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        data = []
        for r in recordings:
            xy, vis, n32 = self._rekaman(r)
            X, valid = _praproses(xy, vis, self.skema)
            y = np.asarray(D.to_eval_rate(r.y, r.fs, FS, is_label=True)).astype(np.float32)
            data.append((X, valid, y))
        # statistik normalisasi: hanya dari pasien latih
        allx = np.concatenate([X[v] for X, v, _ in data])
        self.mu = (allx.mean(0) if self.skema == "piksel" else np.zeros(2)).astype(np.float32)
        self.sd = np.maximum(allx.std(0), 1e-6).astype(np.float32)
        T = []
        for X, valid, y in data:
            Z = _ke_tensor(X, valid, self.mu, self.sd)
            for s in range(0, max(len(y) - WIN, 0) + 1, STRIDE):
                if valid[s:s + WIN].any(-1).mean() >= 0.2 and len(y[s:s + WIN]) == WIN:
                    T.append((Z[s:s + WIN], y[s:s + WIN]))
        Xw = torch.tensor(np.stack([a for a, _ in T])); yw = torch.tensor(np.stack([b for _, b in T]))
        pos = float(yw.sum()); pw = (yw.numel() - pos) / max(pos, 1.0)
        self.net = STGCN().to(self.dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=LR, weight_decay=WD)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=self.dev))
        g = torch.Generator().manual_seed(self.seed)
        for ep in range(self.epochs):
            self.net.train(); perm = torch.randperm(len(Xw), generator=g); tot = 0.0
            for i in range(0, len(perm), BATCH):
                idx = perm[i:i + BATCH]
                xb = Xw[idx].to(self.dev); yb = yw[idx].to(self.dev)
                xb = xb + 0.02 * torch.randn_like(xb) * (xb[..., 2:3] > 0)
                loss = loss_fn(self.net(xb), yb)
                opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
            self.riwayat.append(tot / len(Xw))
            if (ep + 1) % 10 == 0 or ep + 1 == self.epochs:
                print(f"  epoch {ep + 1}/{self.epochs} loss {self.riwayat[-1]:.4f}", flush=True)
        self.n_jendela, self.pos_weight = len(Xw), pw

    @torch.no_grad()
    def predict_proba(self, recording):
        n128 = len(recording.y)
        xy, vis, n32 = self._rekaman(recording)
        X, valid = _praproses(xy, vis, self.skema)
        Z = torch.tensor(_ke_tensor(X, valid, self.mu, self.sd))[None].to(self.dev)
        self.net.eval()
        p32 = torch.sigmoid(self.net(Z))[0].cpu().numpy()
        p = np.repeat(p32, recording.fs // FS)
        if len(p) < n128:
            p = np.concatenate([p, np.full(n128 - len(p), p[-1])])
        return p[:n128].astype(np.float32)

    def card(self):
        npar = sum(p.numel() for p in self.net.parameters()) if self.net is not None else None
        return {
            "nama": self.name, "n_parameter": npar,
            "hiperparameter": {"skema_normalisasi": self.skema, "sendi": V, "kanal_masukan": 3,
                               "blok": 4, "kanal": [32, 32, 64, 64], "kernel_temporal": 9, "dilasi": [1, 1, 2, 4],
                               "partisi": "spatial configuration (akar, sentripetal, sentrifugal)",
                               "jendela_latih": WIN, "langkah_jendela": STRIDE, "batch": BATCH, "lr": LR,
                               "weight_decay": WD, "epoch": self.epochs, "optimizer": "Adam",
                               "loss": "BCE berbobot (pos_weight = n_neg/n_pos pasien latih)",
                               "dropout": DROPOUT, "min_visibility": MIN_VIS, "seed": self.seed},
            "deviasi": [
                "Masukan MediaPipe BlazePose 2D, subset 17 sendi; Yan dkk. (2018) 18 sendi OpenPose, Filtjens dkk. (2022) 25 titik 3D Kinect.",
                "Jaringan jauh lebih kecil (4 blok, 32-64 kanal) dibanding Yan dkk. (9 blok, 64-256 kanal); tanpa tahap penghalusan MS-GCN (setara ST-GCN satu tahap).",
                "Kernel temporal 9 seperti Yan dkk., dengan dilasi (1,1,2,4) tanpa stride; keluaran klasifikasi per sampel (segmentasi), bukan per klip.",
                "Loss: BCE berbobot saja; tanpa truncated MSE (tau=4, lambda=0,15) milik Filtjens dkk.",
                "Adam lr 5e-4 (sama dengan Filtjens dkk.), tetapi 30 epoch (Filtjens 100), batch 32 (Filtjens 16); tanpa penjadwalan lr dan tanpa early stopping.",
                "Kanal masukan ketiga = penanda sendi valid (visibility >= 0,5 dan koordinat terhingga); sendi tidak valid diisi 0 setelah normalisasi.",
                "Statistik normalisasi dari pasien latih; skema 'skala' memakai median panjang torso per rekaman (hanya dari pose rekaman itu sendiri, tanpa label).",
                "Dropout 0,25 pada cabang konvolusi temporal tiap blok (Yan dkk.: 0,5).",
                "Augmentasi hanya derau Gaussian kecil; tanpa flip kiri-kanan atau pemotongan acak.",
                "Tanpa penghalusan probabilitas keluaran (model XGBoost Li memakai rerata bergerak 1 detik).",
                "Dataset berbeda (Figshare, label FoG dari IMU, pasien Parkinson) vs. NTU-RGB+D/Kinetics (Yan) dan Kinect (Filtjens)."]}


def factory(seed: int = 0) -> STGCNFoG:
    return STGCNFoG(seed=seed, skema=os.environ.get("FOG_STGCN_SKEMA", "pinggul"))
