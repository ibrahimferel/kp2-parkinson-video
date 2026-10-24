"""Uji kebocoran: menguji PERILAKU, bukan sekadar membaca kode."""
import copy

import numpy as np
import pytest

from fogbench import splits
from fogbench.preprocess import Normalizer
from fogbench.threshold import pick_threshold


def test_normalisasi_tidak_berubah_oleh_data_uji(recordings):
    """Sisipkan nilai ekstrem ke pasien uji. Parameter normalisasi harus tetap."""
    fold = splits.loso_folds({r.subject for r in recordings})[0]
    train, test = splits.split_recordings(recordings, fold)

    sebelum = Normalizer().fit(train).params()
    rusak = copy.deepcopy(test)
    for r in rusak:
        r.raw6[:] = 1e6                       # data uji dirusak habis-habisan
    sesudah = Normalizer().fit(train).params()

    assert sebelum == sesudah, "parameter normalisasi berubah, berarti data uji ikut terpakai"


def test_normalisasi_menolak_pasien_uji(recordings):
    fold = splits.loso_folds({r.subject for r in recordings})[0]
    train, test = splits.split_recordings(recordings, fold)
    with pytest.raises(ValueError, match="kebocoran"):
        Normalizer().fit(train + test, forbidden_subjects=[fold.test_subject])


def test_ambang_hanya_dari_data_latih(recordings):
    """Ambang yang dihitung dari pasien latih tidak boleh berubah oleh data uji."""
    fold = splits.loso_folds({r.subject for r in recordings})[0]
    train, _ = splits.split_recordings(recordings, fold)
    y = np.concatenate([r.y for r in train])
    p = np.clip(y * 0.7 + np.random.default_rng(1).normal(0, 0.1, len(y)), 0, 1)
    assert pick_threshold(y, p) == pick_threshold(y, p)


def test_fold_menolak_pasien_uji_di_latih():
    with pytest.raises(AssertionError):
        splits.Fold(1, 5, (3, 4, 5))
