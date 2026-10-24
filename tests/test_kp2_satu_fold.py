"""satu_fold (kp2/) harus setara dengan run_loso (fogbench/) dan jalankan_loso harus dapat disambung."""
import json

import numpy as np

from fogbench.evaluate import run_loso
from kp2.satu_fold import buat_jalankan_loso, satu_fold
from models.example_logreg import factory

KUNCI = ("fold", "test_subject", "train_subjects", "fs_eval", "seeds", "threshold")


def _baca(folder, i):
    return json.loads((folder / f"fold_{i:02d}.json").read_text(encoding="utf-8"))


def test_satu_fold_setara_run_loso(recordings, tmp_path):
    resmi, salinan = tmp_path / "resmi", tmp_path / "salinan"
    run_loso(factory, recordings, resmi, "contoh", "sintetis", seeds=[0, 1], verbose=False)
    for sub in range(1, 7):
        satu_fold(factory, recordings, sub, salinan, "contoh", dataset="sintetis", seeds=[0, 1])
    for i in range(1, 7):
        a, b = _baca(resmi, i), _baca(salinan, i)
        for k in KUNCI:
            assert a[k] == b[k], f"fold {i}: kunci {k} berbeda"
        assert np.allclose(a["y_true"], b["y_true"])
        assert np.allclose(a["p_mean"], b["p_mean"], atol=1e-9)
        assert np.allclose(a["p_per_seed"], b["p_per_seed"], atol=1e-9)


def test_jalankan_loso_melewati_fold_yang_sudah_ada(recordings, tmp_path):
    jalan = buat_jalankan_loso(recordings, dataset="sintetis", verbose=False)
    jalan(factory, "contoh", tmp_path, [0])
    sebelum = {p.name: p.stat().st_mtime_ns for p in tmp_path.glob("fold_*.json")}
    assert len(sebelum) == 6
    (tmp_path / "fold_03.json").unlink()                      # simulasi sesi putus di tengah
    jalan(factory, "contoh", tmp_path, [0])
    sesudah = {p.name: p.stat().st_mtime_ns for p in tmp_path.glob("fold_*.json")}
    assert set(sesudah) == set(sebelum)
    assert all(sesudah[n] == sebelum[n] for n in sebelum if n != "fold_03.json"), "fold yang sudah ada ditulis ulang"
