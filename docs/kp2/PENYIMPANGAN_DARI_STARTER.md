# Penyimpangan dari starter (fog-benchmark-starter, commit 3e565f2)

`fogbench/`, `config/protocol.yaml`, `config/datasets.yaml`, `docs/PROTOKOL.md`, dan `docs/FORMAT_HASIL.md` tidak diubah
(checksum di `docs/kp2/fogbench.sha256`). Hal yang ditambahkan atau berbeda:

| Hal | Alasan | Pengaman |
|---|---|---|
| `kp2/satu_fold.py` menyalin isi `run_loso` per fold (aturan 2 starter) | `run_loso` baru menulis berkas setelah seluruh fold selesai; sesi Colab yang putus mengulang dari awal | `tests/test_kp2_satu_fold.py`: keluaran identik dengan `run_loso` pada data sintetis |
| `requirements.txt` digabung (torch, xgboost, mediapipe, opencv, dll.) | Kode KP-2 membutuhkannya; disetujui | uji berat memakai `importorskip` |
| Model membaca pose/fitur dari `FOG_POSE_DIR` dan `FOG_FITUR_DIR`, bukan dari `Recording.raw6` | Metode berbasis video; `Recording` dipakai untuk label dan panjang | tercatat di kartu metode |
| XGBoost Li dijalankan di GPU | Agar waktu latih sebanding dengan ST-GCN (arahan pembimbing) | `perangkat_xgboost` di kartu metode; prediksi dapat sedikit berbeda dari CPU |
| Ambang tetap dari `pick_threshold` (F1-maks pada prediksi pasien latih) | Protokol | Temuan ambang ST-GCN diajukan sebagai *issue* ke pembimbing, bukan dengan mengubah `fogbench/` |

IMU dan fusi dibatalkan; kodenya tidak disertakan.
