"""Uji ringan model KP-2. Dilewati bila torch/xgboost belum terpasang (tidak membutuhkan data pasien)."""
import pytest


def test_li_kartu_metode_mencatat_deviasi_dan_perangkat():
    pytest.importorskip("xgboost")
    import models.li_kinematik_xgb as M
    kartu = M.factory(seed=0).card()
    assert kartu["deviasi"], "kartu metode harus mencantumkan penyimpangan dari makalah acuan"
    assert "perangkat_xgboost" in kartu["hiperparameter"]


def test_stgcn_menyediakan_factory():
    pytest.importorskip("torch")
    import models.stgcn as S
    assert callable(S.factory)
