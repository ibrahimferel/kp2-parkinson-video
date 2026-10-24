> README asli dari https://github.com/bungagana/Fog-benchamark-starter (commit `3e565f2`), disalin tanpa perubahan isi.

# fog-benchmark-starter

Kerangka evaluasi untuk deteksi *freezing of gait* (FoG) pada penyakit Parkinson
dari sensor inersia, dengan validasi per pasien yang tidak membocorkan data uji.

Menyediakan bagian yang tidak berpihak pada metode mana pun: pemuat data,
pembagian *leave-one-subject-out*, normalisasi, pemilihan ambang, metrik, dan
format hasil. Metode deteksi tidak disertakan, dan ditambahkan pengguna di folder
`models/`. Hasil antarmetode dan antarpengguna karena itu dapat dibandingkan
langsung.

## Mengapa

Angka pada makalah deteksi FoG sulit dibandingkan satu sama lain. Penyebab paling
sering adalah pembagian data acak per potongan sinyal, sehingga pasien yang sama
muncul di data latih dan data uji, dan model cukup mengenali pasiennya. Penyebab
yang lebih halus adalah normalisasi atau ambang keputusan yang diam-diam ikut
menghitung data uji. Kerangka ini mencegah keduanya di dalam kode, bukan di dalam
dokumen, dan memakai satu definisi metrik untuk seluruh pengguna.

## Pemasangan

Membutuhkan Python 3.10 atau lebih baru.

```bash
git clone https://github.com/bungagana/Fog-benchamark-starter.git
cd Fog-benchamark-starter
pip install -r requirements.txt
python scripts/selftest.py
```

`selftest` menjalankan seluruh rantai pada data buatan. Angkanya tidak bermakna,
yang diperiksa hanya pemasangan sudah benar.

## Cara pakai

**1. Arahkan ke salinan dataset Anda.** Data tidak disertakan repositori ini.

```bash
cp config/paths.example.yaml config/paths.local.yaml   # lalu isi jalurnya
python scripts/check_data.py --dataset figshare        # verifikasi checksum dan isi
python -m pytest tests/ -q                             # pastikan tidak ada kebocoran
```

**2. Tulis metode** sebagai berkas baru di `models/`, mengikuti
`models/example_logreg.py`.

```python
from models.base import FoGModel

class ModelSaya(FoGModel):
    name = "nama-model"

    def fit(self, recordings):
        """Dilatih dari rekaman pasien latih."""

    def predict_proba(self, recording):
        """Probabilitas FoG per sampel, panjangnya sama dengan recording.y."""

    def card(self):
        """Hiperparameter dan penyimpangan dari makalah acuan."""

def factory(seed=0):
    return ModelSaya()
```

**3. Jalankan dan validasi.** Pembagian fold, pengulangan seed, pemilihan ambang,
dan penulisan hasil ditangani kerangka.

```bash
python scripts/run_experiment.py --model models.model_saya --dataset figshare
python scripts/validate_results.py results/nama-model_figshare --n-fold 35
```

**4. Bandingkan beberapa metode.**

```python
from pathlib import Path
from fogbench import report

print(report.table([Path("results/model_a_figshare"),
                    Path("results/model_b_figshare")]))
print(report.paired_tests(Path("results/model_a_figshare"),
                          [Path("results/model_b_figshare")]))
```

Keluarannya tabel perbandingan, uji Wilcoxon berpasangan dengan koreksi Holm, dan
selang kepercayaan bootstrap pada tingkat pasien.

## Aturan

Melanggarnya tidak membuat kode gagal, tetapi membuat angkanya tidak sebanding
dengan angka pengguna lain.

1. Jangan ubah isi `fogbench/`. Modul di dalamnya adalah definisi protokol. Kalau
   ada kekurangan, buka *issue*.
2. Semua metode lewat antarmuka `models/base.py`. Jangan menulis perulangan
   evaluasi sendiri.
3. Jangan menyentuh data pasien uji di luar `predict_proba`, termasuk untuk
   normalisasi, pemilihan ambang, dan pemilihan hiperparameter.
4. Catat setiap penyimpangan dari makalah acuan pada `card()`.
5. Satu folder hasil untuk satu metode pada satu dataset.
6. Jangan melakukan *commit* data pasien, `config/paths.local.yaml`, atau isi
   `results/`.

## Struktur

| Folder | Isi |
|---|---|
| `config/` | keterangan dataset dan protokol. `paths.local.yaml` bersifat lokal |
| `fogbench/` | infrastruktur evaluasi, tidak untuk diubah pengguna |
| `models/` | metode deteksi, satu berkas per metode |
| `scripts/` | antarmuka baris perintah |
| `tests/` | uji kebocoran, pembagian fold, dan metrik |
| `docs/` | [protokol evaluasi](docs/PROTOKOL.md) dan [format berkas hasil](docs/FORMAT_HASIL.md) |

## Dataset

Data pasien tidak disertakan dan tidak boleh diunggah ke repositori mana pun.
Dataset yang didukung bersifat publik dan diperoleh dari sumber resminya.

| Dataset | Rujukan | Pemuat |
|---|---|---|
| Figshare turning-in-place | De Souza dkk. (2022) | tersedia |
| Daphnet | Bächlin dkk. (2010) | belum |
| FoG-STAR | Borzì dkk. (2026) | belum |

Pemuat baru ditambahkan pada `fogbench/data.py` dengan keluaran yang sama, yaitu
enam kanal mentah dan label per sampel, lalu didaftarkan pada `_READERS`.

## Lisensi

MIT. Lihat [LICENSE](LICENSE).

---

**English.** A subject-independent evaluation harness for freezing-of-gait
detection from inertial sensors. It provides the parts that take no side between
methods: data loading, deterministic leave-one-subject-out splits, train-only
normalisation and threshold selection, shared sample-level and episode-level
metrics, a result-file schema with a validator, and reporting with paired Wilcoxon
tests, Holm correction and subject-level bootstrap confidence intervals. Detection
methods are not included; users add their own under `models/`. Patient data is
never distributed with this repository. Documentation is in Indonesian.
