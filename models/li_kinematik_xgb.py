from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from fogbench import data as D
from models.base import FoGModel

FS = 32
FEATURE_DIR = Path(os.environ.get("FOG_FITUR_DIR", "fitur_li_v2"))
STRIDE_LATIH = 3          # 3 sampel pada 32 Hz = ~0,1 detik, sama dengan langkah jendela Li dkk.
TOP_K = 20
SMOOTH_DETIK = 1.0        # penghalusan probabilitas per rekaman (rerata bergerak terpusat)
_CACHE: dict = {}
# Perangkat XGBoost: "cuda" (GPU, default, sama dengan ST-GCN agar waktu latih sebanding) atau "cpu".
# Ubah lewat variabel lingkungan FOG_XGB_DEVICE.
XGB_DEVICE = os.environ.get("FOG_XGB_DEVICE", "cuda").lower()


def _muat(subject: int, session: int) -> np.ndarray:
    key = (subject, session)
    if key not in _CACHE:
        _CACHE[key] = np.load(FEATURE_DIR / f"PDFE{subject:02d}_{session + 1}.npz")["X"].astype(np.float32)
    return _CACHE[key]


def _selaraskan(X: np.ndarray, n: int) -> np.ndarray:
    """Samakan panjang fitur dengan n sampel 32 Hz: potong, atau isi NaN di ujung."""
    if len(X) >= n:
        return X[:n]
    return np.vstack([X, np.full((n - len(X), X.shape[1]), np.nan, np.float32)])


class LiKinematikXGB(FoGModel):
    name = "li-kinematik-xgb"

    def __init__(self, seed: int = 0, n_pohon: int = 200, kedalaman: int = 4):
        self.seed, self.n_pohon, self.kedalaman = seed, n_pohon, kedalaman
        self.idx = None
        self.model = None

    def _xgb(self, n_pohon, spw):
        kw = dict(n_estimators=n_pohon, max_depth=self.kedalaman, learning_rate=0.1,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                  scale_pos_weight=spw, n_jobs=-1,
                  random_state=self.seed, eval_metric="logloss", verbosity=0)
        if XGB_DEVICE == "cuda":
            import xgboost
            if int(xgboost.__version__.split(".")[0]) >= 2:
                kw.update(tree_method="hist", device="cuda")
            else:
                kw.update(tree_method="gpu_hist")
        else:
            kw.update(tree_method="hist")
        return XGBClassifier(**kw)

    def fit(self, recordings) -> None:
        Xs, ys = [], []
        for r in recordings:
            y32 = D.to_eval_rate(r.y, r.fs, FS, is_label=True)
            X = _selaraskan(_muat(r.subject, r.session), len(y32))
            Xs.append(X[::STRIDE_LATIH]); ys.append(np.asarray(y32)[::STRIDE_LATIH])
        X, y = np.vstack(Xs), np.concatenate(ys).astype(int)
        n_pos = max(int(y.sum()), 1)
        spw = (len(y) - n_pos) / n_pos
        # seleksi fitur HANYA dari pasien latih: peringkat gain XGBoost, ambil TOP_K
        pilih = self._xgb(100, spw).fit(X, y)
        gain = pilih.get_booster().get_score(importance_type="gain")
        urut = sorted(gain, key=gain.get, reverse=True)[:TOP_K]
        self.idx = np.array(sorted(int(k[1:]) for k in urut))
        self.model = self._xgb(self.n_pohon, spw).fit(X[:, self.idx], y)

    def predict_proba(self, recording) -> np.ndarray:
        n128 = len(recording.y)
        n32 = len(D.to_eval_rate(recording.y, recording.fs, FS, is_label=True))
        X = _selaraskan(_muat(recording.subject, recording.session), n32)[:, self.idx]
        p32 = self.model.predict_proba(X)[:, 1]
        if SMOOTH_DETIK > 0:        # hanya memakai probabilitas rekaman ini sendiri, tanpa label
            p32 = pd.Series(p32).rolling(int(SMOOTH_DETIK * FS), center=True, min_periods=1).mean().to_numpy()
        p = np.repeat(p32, recording.fs // FS)
        if len(p) < n128:
            p = np.concatenate([p, np.full(n128 - len(p), p[-1])])
        return p[:n128].astype(np.float32)

    def card(self) -> dict:
        n_simpul = None
        if self.model is not None:
            n_simpul = int(len(self.model.get_booster().trees_to_dataframe()))
        return {
            "nama": self.name,
            "n_parameter": n_simpul,
            "n_parameter_keterangan": "jumlah simpul pohon XGBoost (model akhir, TOP_K fitur)",
            "hiperparameter": {
                "n_pohon": self.n_pohon, "kedalaman_maks": self.kedalaman, "learning_rate": 0.1,
                "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
                "scale_pos_weight": "n_negatif/n_positif dari pasien latih",
                "seleksi_fitur": f"peringkat gain XGBoost (100 pohon) dari pasien latih, ambil {TOP_K}",
                "jendela_detik": 2, "langkah_latih_sampel": STRIDE_LATIH,
                "penghalusan_probabilitas_detik": SMOOTH_DETIK, "seed": self.seed,
                "perangkat_xgboost": XGB_DEVICE, "tree_method": "hist (device=cuda)" if XGB_DEVICE == "cuda" else "hist (CPU)"},
            "deviasi": [
                "Pose dari MediaPipe BlazePose (33 keypoint 2D), bukan OpenPose 25 keypoint seperti Li dkk. (2022).",
                "Hanya 10 keypoint tubuh bawah (pinggul, lutut, pergelangan, tumit, jempol kaki); Li dkk. memakai juga bahu, siku, pergelangan tangan, telinga, leher.",
                "Sinyal: kecepatan, percepatan, sudut lutut, selisih kiri-kanan 5 pasang (x dan y); 32 sinyal. Posisi mentah tidak dipakai sebagai sinyal.",
                "Fitur per jendela 2 detik: 8 domain waktu + 5 domain frekuensi (luas pita 0,5-3 Hz dan 3,5-15 Hz, FI = daya 3-8 Hz / daya 0,5-3 Hz, amplitudo puncak, frekuensi dominan). Daftar fitur Li dkk. lebih panjang (mis. clearance factor, mutual information).",
                "Seleksi fitur: peringkat gain XGBoost dari pasien latih (top 20), bukan forward feature selection seperti Li dkk.",
                "Tidak ada model bertahap jalan/berbalik (multi-stage); satu model FoG untuk seluruh rekaman.",
                "Li dkk. menghapus episode FoG prediksi yang lebih pendek dari kuantil 10% durasi FoG. Di sini diganti penghalusan probabilitas (rerata bergerak 1 detik per rekaman) di dalam predict_proba, karena penghapusan episode butuh ambang yang dipilih oleh penjalan eksperimen.",
                "Normalisasi koordinat: satuan = tinggi crop ROI manual per video (konstan per video, isotropik); Li dkk. memakai kotak pembatas tubuh per frame diperluas 30% dan diskalakan ke tinggi 80.",
                "Laju 32 Hz (Li dkk. 30 Hz). Fitur dihitung tiap sampel 32 Hz; pelatihan memakai langkah 3 sampel (~0,1 detik, sama dengan Li dkk.).",
                "Frame pose yang tidak terdeteksi dibiarkan NaN; XGBoost menanganinya sebagai nilai hilang. Visibility keypoint tidak dipakai untuk menyaring.",
                "Hiperparameter XGBoost tidak disebut di makalah; ditentukan sendiri tanpa penyetelan.",
                f"Pelatihan XGBoost dijalankan di perangkat '{XGB_DEVICE}' (GPU yang sama dengan ST-GCN agar waktu latih sebanding); prediksi dapat sedikit berbeda dari run CPU.",
                "Dataset berbeda: Figshare (berbalik di tempat, label dari IMU) vs. 50 pasien TUG di Li dkk.",
            ]}


def factory(seed: int = 0) -> LiKinematikXGB:
    return LiKinematikXGB(seed=seed)
