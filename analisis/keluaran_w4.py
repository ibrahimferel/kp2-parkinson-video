import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from fogbench import report

OUT_KEL = HASIL / "analisis" / "keluaran"
OUT_KEL.mkdir(parents=True, exist_ok=True)


def pm(x, d=3):
    x = np.asarray(x, float)
    return f"{x.mean():.{d}f} ± {x.std(ddof=1):.{d}f}" if len(x) > 1 else f"{x.mean():.{d}f}"


def metrik_seed(folds):
    """Array (n_seed, 3): AUROC gabungan, AUPRC gabungan, AUROC per pasien -- per seed."""
    hasil = []
    for s in range(len(folds[0]["p_per_seed"])):
        y = np.concatenate([np.asarray(f["y_true"]) for f in folds])
        p = np.concatenate([np.asarray(f["p_per_seed"][s]) for f in folds])
        pas = [roc_auc_score(f["y_true"], f["p_per_seed"][s]) for f in folds if len(np.unique(f["y_true"])) == 2]
        hasil.append((roc_auc_score(y, p), average_precision_score(y, p), np.mean(pas)))
    return np.array(hasil)


def baris(nama, label):
    ms = metrik_seed(FOLDS[nama])
    s = report.summarize(RUNS[nama]); p = s["pooled"]; q = s["per_subject_ringkas"]
    return {"Metode": label, "Seed": ms.shape[0], "Masukan": "video",
            "AUROC": pm(ms[:, 0]), "AUPRC": pm(ms[:, 1]),
            "Sens*": round(p["sensitivitas"], 3), "Spes*": round(p["spesifisitas"], 3),
            "F1*": round(p["f1"], 3), "F1 episode*": round(p["episode_f1"], 3),
            "AUROC per pasien": pm(ms[:, 2]), "SD antar pasien*": round(q["auroc_sd"], 3)}


def simpan(df, nama, judul):
    df.to_csv(OUT_KEL / f"{nama}.csv", index=False)
    (OUT_KEL / f"{nama}.md").write_text(f"### {judul}\n\n" + df.to_markdown(index=False) + "\n", encoding="utf-8")
    print(f"\n{judul}\n" + df.to_string(index=False))


# Keluaran 2: kinerja metode skeleton (fusi dihapus)
ada = lambda k: k in RUNS and k in FOLDS
K2 = [(k, l) for k, l in [("li_1seed", "Fitur kinematik + XGBoost (Li dkk., 2022), 1 seed"),
                          ("li_3seed", "Fitur kinematik + XGBoost (Li dkk., 2022), 3 seed"),
                          ("stgcn_pinggul_1seed", "ST-GCN (Filtjens dkk., 2022), pinggul, 1 seed"),
                          ("stgcn_pinggul_3seed", "ST-GCN (Filtjens dkk., 2022), pinggul, 3 seed"),
                          ("stgcn_skala_1seed", "ST-GCN (Filtjens dkk., 2022), skala tubuh, 1 seed"),
                          ("stgcn_skala_3seed", "ST-GCN (Filtjens dkk., 2022), skala tubuh, 3 seed"),
                          ("stgcn_piksel_1seed", "ST-GCN (Filtjens dkk., 2022), piksel, 1 seed"),
                          ("stgcn_piksel_3seed", "ST-GCN (Filtjens dkk., 2022), piksel, 3 seed")] if ada(k)]
simpan(pd.DataFrame([baris(k, l) for k, l in K2]), "keluaran2_kinerja_metode",
       "Keluaran 2. Kinerja metode skeleton (video saja; fusi dibatalkan)")

# Keluaran 3: kinerja menurut kualitas pose
def auroc_rekaman(nama):
    """AUROC per rekaman per seed (hanya rekaman dengan dua kelas) + kelompok fraksi frame valid."""
    folds = FOLDS[nama]; n_seed = len(folds[0]["p_per_seed"]); baris_ = []
    for fj in folds:
        sl = potong(fj)
        if sl is None:
            raise RuntimeError(f"{nama}: fold {fj['test_subject']} tidak cocok dengan pemotongan rekaman")
        y = np.asarray(fj["y_true"]).astype(int)
        for rek, i, j in sl:
            if 0 < y[i:j].sum() < j - i:
                for s in range(n_seed):
                    baris_.append(dict(rekaman=rek, seed=s,
                                       auroc=roc_auc_score(y[i:j], np.asarray(fj["p_per_seed"][s])[i:j])))
    d = pd.DataFrame(baris_)
    d["kelompok"] = pd.cut(d.rekaman.map(fv), bins=BINS, labels=LAB, right=False)
    return d

K3 = [(k, l) for k, l in [("li_3seed", "AUROC Li (kinematik)"),
                          ("stgcn_pinggul_3seed", "AUROC ST-GCN pinggul"),
                          ("stgcn_skala_3seed", "AUROC ST-GCN skala")] if ada(k)]
base = R[R.model == K3[0][0]]
tab3 = pd.DataFrame({"Kelompok fraksi frame valid": LAB,
                     "Rekaman (semua)": [int((base.kelompok == g).sum()) for g in LAB],
                     "Rekaman (dua kelas, dipakai)": [int(((base.kelompok == g) & base.auroc.notna()).sum()) for g in LAB]})
for k, label in K3:
    d = auroc_rekaman(k)
    per_seed = d.groupby(["seed", "kelompok"], observed=False).auroc.mean().unstack()
    tab3[label] = [pm(per_seed[g].values) if per_seed[g].notna().all() else "-" for g in LAB]
simpan(tab3, "keluaran3_kualitas_pose",
       "Keluaran 3. AUROC rata-rata per rekaman menurut kelompok fraksi frame valid (rerata ± SD antar seed)")

# Keluaran 4: normalisasi koordinat pada ST-GCN
K4 = []
for seed_lab in ("3seed", "1seed"):
    for skema, kunci in [("Koordinat piksel", "piksel"), ("Relatif terhadap pinggul", "pinggul"),
                         ("Dinormalisasi skala tubuh", "skala")]:
        k = f"stgcn_{kunci}_{seed_lab}"
        if ada(k):
            b = baris(k, skema)
            K4.append({"Normalisasi": skema, "Seed": b["Seed"], "AUROC": b["AUROC"],
                       "AUROC per pasien": b["AUROC per pasien"], "F1*": b["F1*"],
                       "Sens*": b["Sens*"], "Spes*": b["Spes*"]})
simpan(pd.DataFrame(K4), "keluaran4_normalisasi",
       "Keluaran 4. Pengaruh normalisasi koordinat keypoint pada ST-GCN")
print("\nUji Wilcoxon + Holm antar skema: lihat blok [G4]/[G4c] pada keluaran analisis_w4/eda_w4.")
print(f"Berkas: {OUT_KEL}")
