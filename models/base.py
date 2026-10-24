from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class FoGModel(ABC):
    name: str = "tanpa-nama"

    @abstractmethod
    def fit(self, recordings: list) -> None:
        """Latih model. HANYA menerima rekaman pasien latih."""

    @abstractmethod
    def predict_proba(self, recording) -> np.ndarray:
        """Probabilitas FoG untuk setiap sampel, panjangnya sama dengan recording.y."""

    def card(self) -> dict:
        """Kartu metode: apa yang Anda pakai dan apa yang menyimpang dari makalah asli."""
        return {"nama": self.name, "n_parameter": None, "hiperparameter": {}, "deviasi": []}
