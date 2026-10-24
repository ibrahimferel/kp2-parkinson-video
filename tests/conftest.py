import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fogbench.data import Recording


@pytest.fixture
def recordings():
    """Data sintetis: 6 pasien, 2 rekaman masing-masing. Tidak perlu data asli."""
    rng = np.random.default_rng(0)
    recs = []
    for sub in range(1, 7):
        for ses in range(2):
            T = 640
            raw6 = rng.normal(0, 1, (T, 6)).astype(np.float32)
            y = np.zeros(T, np.int8)
            y[200:260] = 1
            y[400:430] = 1
            raw6[y == 1] += 3.0
            recs.append(Recording(sub, ses, "sintetis", 64, raw6, y))
    return recs
