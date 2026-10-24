# KP-2: Estimasi Pose dari Video Klinis dan Evaluasi Deteksi FoG Berbasis Skeleton

Kerja Praktik KP-2 (ITS, Laboratorium MCI). Pembimbing: Bunga Laelatul Muna. Pelaksana: Ibrahim Ferel.

Repositori ini memuat kode untuk menilai seberapa bergantung deteksi *freezing of gait* (FoG) pada kualitas pose dari
video klinis, dan membandingkan tiga skema normalisasi koordinat. Evaluasi memakai kerangka
[fog-benchmark-starter](https://github.com/bungagana/Fog-benchamark-starter) (commit `3e565f2`) **tanpa mengubah
`fogbench/`**: validasi leave-one-subject-out, normalisasi dan ambang hanya dari pasien latih, 3 seed (0, 1, 2),
metrik dan format hasil yang sama. Dataset: Figshare turning-in-place (De Souza dkk., 2022).

## Isi

| Folder | Isi |
|---|---|
| `fogbench/`, `config/`, `scripts/`, `tests/`, `docs/PROTOKOL.md`, `docs/FORMAT_HASIL.md` | salinan starter, tidak diubah (`python scripts/cek_fogbench_tak_berubah.py` memeriksanya) |
| `models/stgcn.py` | ST-GCN pada keypoint 2D (Yan dkk., 2018; Filtjens dkk., 2022); skema normalisasi `pinggul`, `skala`, `piksel` lewat `FOG_STGCN_SKEMA` |
| `models/li_kinematik_xgb.py` | fitur kinematik + XGBoost (Li dkk., 2022); XGBoost berjalan di GPU bila ada (`FOG_XGB_DEVICE=cpu` untuk CPU) |
| `kp2/satu_fold.py` | `satu_fold` dan `jalankan_loso`: salinan isi `run_loso` per fold agar dapat disambung setelah sesi putus (diuji setara, `tests/test_kp2_satu_fold.py`) |
| `kp2/w4_runner.py` | antrean 8 run (Li, ST-GCN pinggul/skala/piksel; 1 dan 3 seed), catatan lingkungan, waktu latih per fold, kartu metode |
| `analisis/` | analisis LOSO, EDA per pasien, Keluaran 2/3/4, diagnostik ambang, sensitivitas lag |
| `notebooks/` | alur Colab, output dihapus; `05_week4_eksperimen.ipynb` adalah alur terbaru |
| `docs/kp2/` | alur kerja, penyimpangan dari starter, tabel hasil ringkas, kartu metode |

## Cara menjalankan (Colab, GPU)

1. Unduh dataset ke Drive pribadi dengan `notebooks/01_unduh_figshare.ipynb`.
2. Tahap 1 (ROI, pose MediaPipe BlazePose, resampling 32 Hz, fitur Li): `02_*` dan `03_*`. Menghasilkan `pose_grid32/` dan
   `fitur_li_v2/` di Drive.
3. Eksperimen: `05_week4_eksperimen.ipynb` (isi `REPO_URL` di sel pertama). Pakai satu jenis GPU untuk semua run.

Secara lokal: `pip install -r requirements.txt`, `python -m pytest tests -q`, `python scripts/selftest.py`.

## Data dan keamanan

Video, IMU, pose, fitur, dan berkas hasil per fold **tidak disertakan** (lihat `.gitignore`). Akses data di luar komputer
laboratorium (termasuk Colab) sudah disetujui pembimbing; tautan Drive tidak boleh dibagikan. Yang ada di
`docs/kp2/hasil_ringkas/` hanya tabel ringkasan metrik.

## Lisensi

MIT, mengikuti starter (lihat `LICENSE`).
