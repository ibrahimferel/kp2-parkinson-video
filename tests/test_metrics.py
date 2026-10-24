import numpy as np

from fogbench.metrics import _runs, episode_f1, sample_metrics


def test_deteksi_segmen():
    y = np.array([0, 1, 1, 0, 0, 1, 0])
    assert _runs(y) == [(1, 3), (5, 6)]


def test_episode_terdeteksi_penuh():
    y = np.array([0, 0, 1, 1, 1, 1, 0, 0])
    r = episode_f1(y, y)
    assert r["episode_f1"] == 1.0 and r["n_episode_true"] == 1


def test_episode_tumpang_tindih_kurang_dari_setengah():
    y = np.array([0, 1, 1, 1, 1, 0])
    pred = np.array([0, 1, 0, 0, 0, 0])          # hanya 1 dari 4 sampel
    assert episode_f1(y, pred)["episode_recall"] == 0.0


def test_auroc_sempurna():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert sample_metrics(y, p, 0.5)["auroc"] == 1.0


def test_satu_kelas_menghasilkan_nan():
    y = np.zeros(10, int)
    assert np.isnan(sample_metrics(y, np.full(10, 0.3), 0.5)["auroc"])
