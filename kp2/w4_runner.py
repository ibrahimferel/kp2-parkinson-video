# Week-4: struktur folder, catatan lingkungan, antrean LOSO dengan log waktu latih
import os, sys, json, time, shutil, platform, subprocess, importlib, datetime
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(os.environ.get("KP_BASE", "/content/drive/MyDrive/KP-TA MCI"))
W2, W3, W4 = BASE / "05. Week 2", BASE / "06. Week 3", BASE / "07. Week 4"
HASIL, LOG, KARTU = W4 / "hasil", W4 / "log", W4 / "kartu_metode"
KODE = Path(os.environ.get("KP_KODE", W4 / "kode_model"))
ANALISIS = HASIL / "analisis"

# Urutan: Li (XGBoost GPU) dulu, lalu ST-GCN (GPU); semua run di GPU yang sama. Tiap entri dijalankan SATU per satu (jangan paralel: mengacaukan waktu latih).
# 1 seed = seed 0 saja; 3 seed = seed 0,1,2. Keduanya dilatih terpisah, sesuai daftar permintaan Bunga.
ANTREAN = [
    # (folder_hasil,           modul,                skema,     seeds)
    ("li_1seed",               "li_kinematik_xgb",   None,      [0]),
    ("li_3seed",               "li_kinematik_xgb",   None,      [0, 1, 2]),
    ("stgcn_pinggul_1seed",    "stgcn",              "pinggul", [0]),
    ("stgcn_pinggul_3seed",    "stgcn",              "pinggul", [0, 1, 2]),
    ("stgcn_skala_1seed",      "stgcn",              "skala",   [0]),
    ("stgcn_skala_3seed",      "stgcn",              "skala",   [0, 1, 2]),
    ("stgcn_piksel_1seed",     "stgcn",              "piksel",  [0]),
    ("stgcn_piksel_3seed",     "stgcn",              "piksel",  [0, 1, 2]),
]


# 0. struktur folder
def siapkan_folder():
    """Buat struktur Week 4 dan salin kode model dari Week 3 (Week 3 tidak disentuh)."""
    for d in (HASIL, ANALISIS, LOG, KODE, KARTU, HASIL / "_sebagian"):
        d.mkdir(parents=True, exist_ok=True)
    sumber = W3 / "kode_model"
    for f in ("stgcn.py", "li_kinematik_xgb.py"):          # IMU & fusi dibatalkan -> tidak disalin
        if (sumber / f).exists() and not (KODE / f).exists():
            shutil.copy2(sumber / f, KODE / f)
    print("Folder Week 4:", W4)
    for d in (HASIL, LOG, KODE, KARTU):
        print("  ", d.name, "->", len(list(d.iterdir())), "item")


def pasang_kode():
    """Salin stgcn.py & li_kinematik_xgb.py dari Week 4 ke paket `models` yang dipakai kerangka (agar versi W4 yang berjalan)."""
    import models
    assert "DROPOUT" in (KODE / "stgcn.py").read_text(), "stgcn.py di kode_model belum versi tambalan (DROPOUT)"
    dst = Path(models.__file__).parent
    for f in ("stgcn.py", "li_kinematik_xgb.py"):
        shutil.copy2(KODE / f, dst / f)
    for m in [m for m in sys.modules if m.startswith("models.") and m != "models.base"]:
        sys.modules.pop(m)                                   # paksa impor ulang (konstanta env dibaca saat impor)
    print("Kode terpasang di", dst)


def lokalkan_data():
    """Salin folder pose/fitur dari Drive ke disk lokal Colab agar waktu latih tidak tercampur latensi Drive."""
    for var in ("FOG_POSE_DIR", "FOG_FITUR_DIR"):
        v = os.environ.get(var)
        if not v or not Path(v).exists():
            print(f"[!] {var} = {v!r} tidak ditemukan -> cek dulu sebelum menjalankan antrean"); continue
        p = Path(v).resolve()
        if str(p).startswith("/content/drive"):
            dst = Path("/content/data_lokal") / p.name
            if not dst.exists():
                t = time.time(); shutil.copytree(p, dst); print(f"{var}: disalin ke {dst} ({time.time()-t:.0f} s)")
            os.environ[var] = str(dst)
        print(f"{var} = {os.environ[var]}")


# 1. catatan lingkungan
def catat_lingkungan(label="mulai"):
    """Simpan perangkat keras/versi pustaka. Wajib sama untuk seluruh run yang waktunya dibandingkan."""
    info = dict(waktu=datetime.datetime.now().isoformat(timespec="seconds"), label=label,
                python=platform.python_version(), cpu_logis=os.cpu_count(),
                ram_gb=None, gpu=None, cuda=None, torch=None, xgboost=None, numpy=np.__version__,
                env={k: os.environ.get(k) for k in ("FOG_POSE_DIR", "FOG_FITUR_DIR", "FOG_STGCN_SKEMA", "FOG_STGCN_EPOCHS")})
    try:
        info["ram_gb"] = round(int([l for l in open("/proc/meminfo") if l.startswith("MemTotal")][0].split()[1]) / 1e6, 1)
    except Exception: pass
    try:
        import torch
        info["torch"] = torch.__version__; info["cuda"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "TIDAK ADA GPU"
    except Exception: pass
    try:
        import xgboost; info["xgboost"] = xgboost.__version__
    except Exception: pass
    try:
        info["cpu_model"] = [l.split(":")[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
    except Exception: pass
    (LOG / f"lingkungan_{label}.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    return info


# 2. antrean
def _fold_files(folder):
    return sorted(Path(folder).glob("fold_*.json"))


def _sync():
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.synchronize()
    except Exception: pass


def jalankan_antrean(jalankan_loso, pilih=None, ulang=False):
    """Jalankan ANTREAN berurutan. pilih=['stgcn_pinggul_1seed',...] untuk sebagian.
    - Run yang sudah 35 fold dilewati.
    - Run yang baru separuh DILANJUTKAN (jalankan_loso melewati fold yang berkasnya ada), tetapi hanya jika GPU sama dengan
      segmen sebelumnya (tercatat di log/jejak_run.csv); jika beda -> berhenti, pakai ulang=True untuk mengulang dari awal.
    - ulang=True: folder lama dipindah ke hasil/_sebagian lalu dijalankan dari awal."""
    siapkan_folder(); pasang_kode(); lokalkan_data()
    lingk = catat_lingkungan("antrean_" + datetime.datetime.now().strftime("%m%d_%H%M"))
    print("GPU:", lingk["gpu"], "| CPU:", lingk.get("cpu_model"), "x", lingk["cpu_logis"], "| RAM(GB):", lingk["ram_gb"])
    jejak = LOG / "jejak_run.csv"
    for nama, modul, skema, seeds in ANTREAN:
        if pilih and nama not in pilih: continue
        out = HASIL / nama; n = len(_fold_files(out)) if out.exists() else 0
        if n >= 35 and not ulang:
            print(f"[lewati] {nama}: 35 fold sudah ada"); continue
        if n > 0 and ulang:
            tujuan = HASIL / "_sebagian" / f"{nama}_{datetime.datetime.now().strftime('%m%d_%H%M')}"
            shutil.move(str(out), str(tujuan)); print(f"[{nama}] {n} fold lama dipindah ke {tujuan.name}"); n = 0
            if jejak.exists():      # baris jejak run lama ikut dipindah agar wall-clock tidak tercampur
                J = pd.read_csv(jejak); lama = J["run"] == nama
                if lama.any():
                    J[lama].to_csv(LOG / "jejak_run_sebagian.csv", mode="a", header=not (LOG / "jejak_run_sebagian.csv").exists(), index=False)
                    J[~lama].to_csv(jejak, index=False)
        if skema: os.environ["FOG_STGCN_SKEMA"] = skema
        mod = importlib.import_module(f"models.{modul}"); importlib.reload(mod)
        label = "li-kinematik-xgb" if modul.startswith("li_") else f"stgcn-{skema}"
        gpu_awal = catat_lingkungan("run_" + nama)["gpu"]
        if gpu_awal == "TIDAK ADA GPU":
            raise RuntimeError("Runtime tanpa GPU: ganti ke GPU (mis. L4) dulu, dan pakai tipe yang SAMA untuk semua run (Li dan ST-GCN)")
        if n > 0 and jejak.exists():
            J = pd.read_csv(jejak); gpu_lama = set(J.loc[J["run"] == nama, "gpu"].dropna())
            if gpu_lama and gpu_awal not in gpu_lama:
                raise RuntimeError(f"{nama}: dilanjutkan di GPU '{gpu_awal}' padahal segmen sebelumnya {gpu_lama} -> waktu latih tidak sebanding. "
                                   f"Pakai GPU yang sama atau jalankan_antrean(..., pilih=['{nama}'], ulang=True)")
        print(f"\n=== {nama} | seeds={seeds} | fold sudah ada: {n}/35 | mulai {datetime.datetime.now():%H:%M:%S} ===")
        _sync(); t0 = time.perf_counter(); mulai = datetime.datetime.now().isoformat(timespec="seconds")
        jalankan_loso(mod.factory, label, out, seeds)
        _sync(); wall = time.perf_counter() - t0
        baris = _baris_jejak(nama, modul, skema, seeds, out, wall, mulai, gpu_awal); baris["fold_sebelum_segmen"] = n
        pd.DataFrame([baris]).to_csv(jejak, mode="a", header=not jejak.exists(), index=False)
        print(f"=== {nama} selesai: {baris['n_fold']} fold | wall-clock segmen ini {wall/60:.1f} menit | "
              f"jumlah train_time_s (seluruh fold) {baris['jumlah_train_time_s']/60:.1f} menit | GPU {gpu_awal} ===")
    tulis_kartu_metode(); ringkas_waktu()


def _baris_jejak(nama, modul, skema, seeds, out, wall, mulai, gpu):
    fs = [json.loads(f.read_text()) for f in _fold_files(out)]
    tt = np.array([f.get("train_time_s", np.nan) for f in fs], float)
    return dict(run=nama, modul=modul, skema=skema, seeds=str(seeds), n_fold=len(fs), mulai=mulai, gpu=gpu,
                wall_clock_s=round(wall, 1), jumlah_train_time_s=round(float(np.nansum(tt)), 1),
                train_per_wall=round(float(np.nansum(tt)) / wall, 2) if wall > 0 else None,
                median_train_time_s_per_fold=round(float(np.nanmedian(tt)), 1) if len(tt) else None)


# 3. pelaporan waktu latih + kartu metode
def ringkas_waktu():
    """Tabel waktu latih per fold dan ringkasan per run -> log/waktu_latih_per_fold.csv, log/ringkasan_waktu.csv."""
    rows = []
    for nama, *_ in ANTREAN:
        for f in _fold_files(HASIL / nama):
            d = json.loads(f.read_text())
            rows.append(dict(run=nama, fold=d["fold"], pasien_uji=d["test_subject"], n_seed=len(d["seeds"]),
                             train_time_s=d.get("train_time_s"), infer_time_ms=d.get("infer_time_ms")))
    if not rows: print("belum ada hasil"); return None
    T = pd.DataFrame(rows); T.to_csv(LOG / "waktu_latih_per_fold.csv", index=False)
    S = []
    for nama, g in T.groupby("run", sort=False):
        g = g.sort_values("fold"); t = g["train_time_s"].astype(float)
        S.append(dict(run=nama, n_fold=len(g), n_seed=int(g["n_seed"].iloc[0]), total_menit=t.sum() / 60,
                      rerata_per_fold_s=t.mean(), median_per_fold_s=t.median(), sd_per_fold_s=t.std(ddof=1),
                      min_s=t.min(), maks_s=t.max(), fold_pertama_s=t.iloc[0],
                      median_tanpa_fold_pertama_s=t.iloc[1:].median() if len(t) > 1 else np.nan,
                      infer_median_ms=g["infer_time_ms"].astype(float).median()))
    S = pd.DataFrame(S)
    # rasio 3 seed / 1 seed: harus ~3 (train_time_s = jumlah seluruh seed)
    r = {}
    for nama in S["run"]:
        if nama.endswith("_3seed") and nama.replace("_3seed", "_1seed") in set(S["run"]):
            a = S.set_index("run"); r[nama] = a.loc[nama, "median_per_fold_s"] / a.loc[nama.replace("_3seed", "_1seed"), "median_per_fold_s"]
    S["rasio_3seed_per_1seed"] = S["run"].map(r)
    S["waktu_latih_total_jam"] = S["total_menit"] / 60
    jj = LOG / "jejak_run.csv"
    if jj.exists():
        W = pd.read_csv(jj).groupby("run")["wall_clock_s"].sum() / 3600
        S["wall_clock_total_jam"] = S["run"].map(W)
    S.round(2).to_csv(LOG / "ringkasan_waktu.csv", index=False)
    print(S.round(2).to_string(index=False))
    print("\nCatatan: train_time_s (fogbench satu_fold) = JUMLAH waktu fit seluruh seed pada fold itu; fit model tambahan untuk ambang "
          "(1 model, seed pertama) dan prediksi TIDAK dihitung di dalamnya. wall_clock_total_jam = waktu nyata run (termasuk fit ambang, "
          "prediksi, penulisan berkas). Laporkan keduanya dan jelaskan definisinya.")
    jj = LOG / "jejak_run.csv"
    if jj.exists():
        J = pd.read_csv(jj); print("\nGPU yang dipakai per run:\n", J[["run", "gpu", "wall_clock_s"]].to_string(index=False))
        if J["gpu"].nunique() > 1: print("\n[!] GPU TIDAK SERAGAM antar run -> waktu ST-GCN tidak sebanding; ulangi di tipe GPU yang sama")
    return S


def tulis_kartu_metode():
    """Satu kartu metode (.md) per run dari model_card fold pertama (arsitektur, parameter, hiperparameter, deviasi)."""
    KARTU.mkdir(exist_ok=True)
    for nama, *_ in ANTREAN:
        fl = _fold_files(HASIL / nama)
        if not fl: continue
        d = json.loads(fl[0].read_text()); c = d.get("model_card", {})
        md = [f"# Kartu metode: {nama}", f"- Model: {d.get('model')}", f"- Parameter: {c.get('n_parameter')}",
              f"- Seed: {d.get('seeds')}", "", "## Hiperparameter"]
        md += [f"- {k}: {v}" for k, v in (c.get("hiperparameter") or {}).items()]
        md += ["", "## Penyimpangan dari makalah acuan"] + [f"- {x}" for x in (c.get("deviasi") or [])]
        (KARTU / f"{nama}.md").write_text("\n".join(md), encoding="utf-8")
    print("Kartu metode ditulis ke", KARTU)


print("w4_runner siap. Urutan: siapkan_folder() -> (unggah kode ke kode_model/) -> jalankan_antrean(jalankan_loso)")
