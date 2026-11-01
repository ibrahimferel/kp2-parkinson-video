# Format berkas hasil

Satu berkas JSON per fold, diberi nama `fold_01.json` sampai `fold_35.json`, di
dalam satu folder per model. Format ini yang membuat hasil Anda dapat digabungkan
dengan hasil orang lain.

| Kunci | Isi |
|---|---|
| `model` | nama model |
| `dataset` | nama dataset |
| `fold` | nomor fold |
| `test_subject` | nomor pasien uji |
| `train_subjects` | daftar nomor pasien latih |
| `fs_eval` | laju cuplik penilaian, yaitu 32 |
| `threshold` | ambang keputusan, dipilih dari pasien latih |
| `y_true` | label per sampel pada 32 Hz, hanya berisi 0 dan 1 |
| `p_mean` | probabilitas per sampel, rerata antar seed |
| `p_per_seed` | probabilitas per sampel untuk setiap seed |
| `seeds` | daftar seed yang dipakai |
| `train_time_s` | waktu pelatihan fold ini, detik |
| `infer_time_ms` | waktu inferensi satu rekaman, milidetik |
| `model_card` | kartu metode: nama, jumlah parameter, hiperparameter, daftar deviasi |

Berkas ditulis otomatis oleh penjalan eksperimen, jadi Anda tidak perlu membuatnya
sendiri. Sebelum hasil dibagikan, jalankan pemeriksa:

```bash
python scripts/validate_results.py results/nama-model_figshare --n-fold 35
```

Pemeriksa akan menolak kalau jumlah fold kurang, panjang probabilitas tidak sama
dengan panjang label, ada nilai kosong, probabilitas di luar rentang 0 sampai 1,
atau pasien uji ternyata ikut di daftar pasien latih.

## Kartu metode

Isi `model_card` adalah bagian terpenting dari pelaporan. Cantumkan setiap penyimpangan dari
makalah asli beserta alasannya, misalnya hiperparameter yang tidak disebutkan di
makalah dan Anda tentukan sendiri, atau lapisan yang Anda sederhanakan. Menyimpang
itu wajar. Menyimpang tanpa mencatatnya membuat hasil tidak dapat direproduksi.
