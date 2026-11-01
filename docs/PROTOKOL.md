# Protokol evaluasi bersama

Seluruh eksperimen pada kerangka ini memakai aturan di bawah ini. Hasil yang
tidak mengikutinya tidak sebanding dengan hasil lain yang memakai kerangka ini.

| Aspek | Ketentuan |
|---|---|
| Validasi | Leave-one-subject-out. Satu pasien menjadi data uji, sisanya data latih, diulang untuk setiap pasien |
| Satuan penilaian | Per sampel pada 32 Hz. Prediksi model, apa pun bentuk window-nya, diubah menjadi probabilitas pada setiap sampel 32 Hz |
| Normalisasi | Rerata dan simpangan baku dihitung hanya dari pasien latih pada setiap fold |
| Ambang keputusan | Dipilih dari pasien latih, yaitu yang memaksimumkan F1. Tidak pernah dari pasien uji |
| Pengulangan | Tiga seed acak, yaitu 0, 1, dan 2. Probabilitas dirata-ratakan antar seed |
| Metrik | AUROC dan AUPRC gabungan, AUROC per pasien dengan simpangan baku, sensitivitas, spesifisitas, F1, F1 tingkat episode |
| Episode | Sebuah episode dinyatakan terdeteksi bila prediksi menutupi minimal 50 persen durasinya |
| Statistik | Uji Wilcoxon berpasangan pada AUROC per pasien dengan koreksi Holm, dan selang kepercayaan bootstrap pada tingkat pasien |
| Kesetiaan pada makalah | Arsitektur, masukan, panjang window, dan pengaturan pelatihan mengikuti makalah asli. Setiap penyimpangan dicatat pada kartu metode |

## Mengapa leave-one-subject-out

Kesalahan paling umum pada makalah deteksi FoG adalah membagi data secara acak per
potongan sinyal. Akibatnya potongan dari pasien yang sama muncul di data latih dan
data uji sekaligus, sehingga model cukup mengenali pasiennya, bukan gejalanya.
Angkanya terlihat tinggi tetapi tidak berlaku untuk pasien baru.

## Mengapa ambang dari pasien latih

Ambang keputusan adalah parameter yang dipelajari dari data. Kalau dipilih dengan
melihat hasil pada pasien uji, Anda sudah memakai jawaban untuk menentukan caranya
menjawab. Kenaikan angka yang muncul dari situ tidak nyata.

## Metrik per pasien

Metrik gabungan dihitung dengan menyatukan seluruh sampel dari semua pasien,
sehingga pasien dengan rekaman panjang atau banyak episode mendominasi. Metrik per
pasien menghitung satu nilai untuk setiap pasien lalu merata-ratakannya, sehingga
memperlihatkan apakah model bekerja merata. Keduanya wajib dilaporkan. Pasien yang
tidak memiliki episode FoG tidak punya nilai AUROC dan dikeluarkan dari rerata per
pasien, tetapi tetap masuk metrik gabungan.
