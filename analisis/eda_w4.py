# Sel G: EDA hasil 35 fold (setelah Sel A; membaca berkas fold di Drive, bukan teks output Sel F)
import json, numpy as np, pandas as pd
import matplotlib; import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
from fogbench import report, splits, data as D

URUT = ["li_1seed", "li_3seed", "stgcn_pinggul_1seed", "stgcn_pinggul_3seed", "stgcn_skala_1seed", "stgcn_piksel_1seed", "stgcn_skala_3seed", "stgcn_piksel_3seed"]
RUNS = {k: HASIL/k for k in URUT if (HASIL/k).exists() and len(list((HASIL/k).glob("fold_*.json"))) >= 35}
OUT_AN = HASIL/"analisis"; OUT_AN.mkdir(exist_ok=True)
print("Folder dipakai:", list(RUNS))
FOLDS = {k: report.load_run(v) for k, v in RUNS.items()}
nan = float("nan")

# G1. tabel per pasien per model
rows = []
for nama, fl in FOLDS.items():
    for f in fl:
        y = np.asarray(f["y_true"]).astype(int); p = np.asarray(f["p_mean"]); thr = f["threshold"]; dua = y.min() != y.max()
        pred = p >= thr
        rows.append(dict(model=nama, pasien=f["test_subject"], n=len(y), prevalensi=y.mean(),
                         n_episode=int((np.diff(np.r_[0, y]) == 1).sum()),
                         auroc=roc_auc_score(y, p) if dua else nan, auprc=average_precision_score(y, p) if dua else nan,
                         ambang=thr, p_rata=p.mean(), p_rata_FoG=p[y == 1].mean() if y.any() else nan,
                         p_rata_nonFoG=p[y == 0].mean() if (y == 0).any() else nan,
                         sens=(pred & (y == 1)).sum() / max((y == 1).sum(), 1) if y.any() else nan,
                         spes=(~pred & (y == 0)).sum() / max((y == 0).sum(), 1) if (y == 0).any() else nan,
                         waktu_latih_s=f["train_time_s"]))
T = pd.DataFrame(rows); T.to_csv(OUT_AN/"eda_per_pasien_per_model.csv", index=False)
E = T.dropna(subset=["auroc"])                     # 26 pasien yang punya dua kelas
print("\n[G1] Pasien yang tidak punya FoG (tidak ikut AUROC per pasien):",
      sorted(T[(T.model == list(RUNS)[0]) & T.auroc.isna()].pasien.tolist()))

# G2. ringkasan sebaran AUROC per pasien
r = E.groupby("model").auroc.agg(n="count", rerata="mean", SD="std", median="median", min="min", maks="max",
                                 kurang_dari_0_5=lambda s: int((s < 0.5).sum()), lebih_dari_0_7=lambda s: int((s > 0.7).sum()))
print("\n[G2] Sebaran AUROC per pasien"); print(r.reindex(list(RUNS)).round(3).to_string())

# G3. apakah dua metode gagal pada pasien yang sama?
W = E.pivot(index="pasien", columns="model", values="auroc")
W.round(4).to_csv(OUT_AN/"eda_auroc_pasien_lebar.csv")
if {"li_3seed", "stgcn_pinggul_3seed"} <= set(W.columns):
    a, b = "li_3seed", "stgcn_pinggul_3seed"
elif {"li_1seed", "stgcn_pinggul_1seed"} <= set(W.columns):
    a, b = "li_1seed", "stgcn_pinggul_1seed"
else:
    a = b = None
if a:
    rho = spearmanr(W[a], W[b])[0]
    sulit = W[(W[a] < 0.5) & (W[b] < 0.5)].index.tolist()
    beda = (W[a] - W[b]).abs().sort_values(ascending=False).head(5).round(3)
    print(f"\n[G3] Korelasi Spearman AUROC per pasien {a} vs {b}: {rho:.3f}  (tinggi = gagal/berhasil di pasien yang sama)")
    print("     Pasien sulit (AUROC < 0,5 pada keduanya):", sulit)
    print("     5 pasien dengan selisih terbesar:"); print(beda.to_string())

# G4. RQ2: skema normalisasi
if {"stgcn_pinggul_1seed", "stgcn_skala_1seed", "stgcn_piksel_1seed"} <= set(RUNS):
    print("\n[G4] RQ2: skema normalisasi, 1 seed, referensi = pinggul (selisih = pinggul - pembanding; Holm untuk 2 uji)")
    print(report.paired_tests(RUNS["stgcn_pinggul_1seed"], [RUNS["stgcn_skala_1seed"], RUNS["stgcn_piksel_1seed"]]))
    s = E[E.model.isin(["stgcn_pinggul_1seed", "stgcn_skala_1seed", "stgcn_piksel_1seed"])]
    print(s.groupby("model")[["auroc", "auprc"]].agg(["mean", "std"]).round(3).to_string())
    print("Skala vs pinggul per pasien: skala lebih tinggi pada",
          int((W["stgcn_skala_1seed"] > W["stgcn_pinggul_1seed"]).sum()), "dari", len(W), "pasien")
if {"stgcn_pinggul_3seed", "stgcn_skala_3seed", "stgcn_piksel_3seed"} <= set(RUNS):
    print("\n[G4c] RQ2 pada 3 seed (referensi = pinggul 3 seed; Holm untuk 2 uji)")
    print(report.paired_tests(RUNS["stgcn_pinggul_3seed"], [RUNS["stgcn_skala_3seed"], RUNS["stgcn_piksel_3seed"]]))
    print(E[E.model.isin(["stgcn_pinggul_3seed", "stgcn_skala_3seed", "stgcn_piksel_3seed"])].groupby("model").auroc.agg(["mean", "std", "median"]).round(3).to_string())
if "stgcn_pinggul_3seed" in RUNS:
    print("\n[G4b] Pinggul 3 seed vs 1 seed (mengukur seberapa goyang 1 seed):")
    print(report.paired_tests(RUNS["stgcn_pinggul_3seed"], [RUNS["stgcn_pinggul_1seed"]]))

# G5. goyangan antar seed (hanya folder 3 seed)
for k in [k for k in RUNS if k.endswith("3seed")]:
    sd = []
    for f in FOLDS[k]:
        y = np.asarray(f["y_true"]).astype(int)
        if y.min() == y.max(): continue
        sd.append(np.std([roc_auc_score(y, ps) for ps in f["p_per_seed"]], ddof=1))
    print(f"\n[G5] {k}: SD AUROC antar seed per pasien -> rerata {np.mean(sd):.3f} | maks {np.max(sd):.3f}")

# G6. faktor pasien: prevalensi dan jumlah episode
print("\n[G6] Spearman antara metrik per pasien dan prevalensi / jumlah episode FoG")
for k in RUNS:
    d = E[E.model == k]
    t = T[T.model == k]
    print(f"  {k:22s} AUROC~prevalensi {spearmanr(d.auroc, d.prevalensi)[0]:+.2f} | AUROC~n_episode {spearmanr(d.auroc, d.n_episode)[0]:+.2f}"
          f" | rerata prob.~prevalensi (35 pasien) {spearmanr(t.p_rata, t.prevalensi)[0]:+.2f}")

# G7. kualitas pose per rekaman (RQ1, awal)
META = W2/"metadata_kualitas_pose.csv"
if META.exists():
    m = pd.read_csv(META); m["rekaman"] = m["Rekaman"].str.replace(r"\.[A-Za-z0-9]+$", "", regex=True)
    fv = dict(zip(m.rekaman, m.fraksi_valid))
    BINS = [-0.001, 0.25, 0.50, 0.80, 1.0001]; LAB = ["<0,25", "0,25-0,50", "0,50-0,80", ">=0,80"]
    def potong(fj):
        fold = next(f for f in splits.loso_folds({r.subject for r in recs}) if f.test_subject == fj["test_subject"])
        _, test = splits.split_recordings(recs, fold)
        ys = [np.asarray(D.to_eval_rate(r.y, r.fs, 32, is_label=True)).astype(int) for r in test]
        y = np.concatenate(ys)
        if len(y) != len(fj["y_true"]) or not np.array_equal(y, np.asarray(fj["y_true"]).astype(int)):
            return None
        out, i = [], 0
        for r, yy in zip(test, ys):
            out.append((f"PDFE{r.subject:02d}_{r.session + 1}", i, i + len(yy))); i += len(yy)
        return out
    rr = []; gagal = 0
    for nama, fl in FOLDS.items():
        for fj in fl:
            sl = potong(fj)
            if sl is None: gagal += 1; continue
            y = np.asarray(fj["y_true"]).astype(int); p = np.asarray(fj["p_mean"])
            for rek, i, j in sl:
                rr.append(dict(model=nama, rekaman=rek, frac_valid=fv.get(rek, nan), n_FoG=int(y[i:j].sum()), n=j - i,
                               auroc=roc_auc_score(y[i:j], p[i:j]) if 0 < y[i:j].sum() < j - i else nan,
                               y=y[i:j], p=p[i:j]))
    R = pd.DataFrame(rr); print(f"\n[G7] rekaman terpotong dengan benar; fold gagal dicocokkan: {gagal}")
    R["kelompok"] = pd.cut(R.frac_valid, bins=BINS, labels=LAB, right=False)
    base = R[R.model == list(RUNS)[0]]
    print("Jumlah rekaman per kelompok (semua / punya dua kelas):")
    print(pd.DataFrame({"semua": base.groupby("kelompok", observed=False).size(),
                        "dua_kelas": base.dropna(subset=["auroc"]).groupby("kelompok", observed=False).size()}).to_string())
    tab = []
    for k in RUNS:
        for g in LAB:
            s = R[(R.model == k) & (R.kelompok == g)]
            yy = np.concatenate(s.y.tolist()) if len(s) else np.array([0]); pp = np.concatenate(s.p.tolist()) if len(s) else np.array([0.])
            tab.append(dict(model=k, kelompok=g, n_rekaman_dua_kelas=int(s.auroc.notna().sum()),
                            auroc_gabungan=roc_auc_score(yy, pp) if yy.min() != yy.max() else nan,
                            auroc_rerata_per_rekaman=s.auroc.mean()))
    G = pd.DataFrame(tab); G.round(3).to_csv(OUT_AN/"eda_kualitas_pose.csv", index=False)
    print(G.pivot(index="model", columns="kelompok", values="auroc_gabungan")[LAB].reindex(list(RUNS)).round(3).to_string())
    print("\nAUROC rata-rata per rekaman (hanya rekaman dua kelas) dan jumlahnya:")
    print(G.pivot(index="model", columns="kelompok", values="auroc_rerata_per_rekaman")[LAB].reindex(list(RUNS)).round(3).to_string())
    print(G[G.model == list(RUNS)[0]].set_index("kelompok").n_rekaman_dua_kelas.reindex(LAB).to_string())
    print("(AUROC gabungan per kelompok mencampur pasien; baca bersama jumlah rekaman. Ini EDA awal, bukan hasil resmi RQ1.)")
    d = R[R.model == list(RUNS)[0]].dropna(subset=["auroc", "frac_valid"])
    print("Spearman AUROC per rekaman vs frac_valid (model pertama):", round(spearmanr(d.auroc, d.frac_valid)[0], 3), "| n =", len(d))
else:
    print("\n[G7] dilewati: metadata_kualitas_pose.csv tidak ditemukan di", META)

# G9. apakah selisih antar metode berasal dari pasien dengan sedikit FoG?
if a:
    info = T[T.model == a].set_index("pasien")[["n", "prevalensi", "n_episode"]]
    info["n_FoG"] = (info.n * info.prevalensi).round().astype(int)
    X9 = info.join(W[[a, b]]).dropna().sort_values("n_FoG")
    X9["selisih_abs"] = (X9[a] - X9[b]).abs()
    print("\n[G9] Pasien dua kelas, diurut dari sampel FoG paling sedikit:"); print(X9.round(3).to_string())
    print("Spearman |selisih AUROC| vs jumlah sampel FoG:", round(spearmanr(X9.selisih_abs, X9.n_FoG)[0], 3),
          "(negatif = selisih besar terjadi pada pasien dengan sedikit FoG)")

# G8. gambar
if a:
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    ax[0].scatter(W[a], W[b], color="#2a78d6", s=34); ax[0].plot([0, 1], [0, 1], color="#999", lw=1)
    for pas in W.index: ax[0].annotate(str(pas), (W.loc[pas, a], W.loc[pas, b]), fontsize=7, xytext=(3, 2), textcoords="offset points")
    ax[0].set(xlabel=f"AUROC {a}", ylabel=f"AUROC {b}", title="Per pasien: dua metode", xlim=(0.2, 1.05), ylim=(0.2, 1.05))
    for k, c in zip((a, b), ("#2a78d6", "#eb6834")):
        d = E[E.model == k]; ax[1].scatter(d.prevalensi, d.auroc, color=c, s=30, label=k, alpha=.85)
    ax[1].axhline(0.5, color="#999", lw=1); ax[1].set(xlabel="prevalensi FoG pasien", ylabel="AUROC", title="AUROC vs prevalensi"); ax[1].legend(fontsize=8)
    ks = [k for k in ("stgcn_pinggul_3seed", "stgcn_skala_3seed", "stgcn_piksel_3seed") if k in RUNS]
    if len(ks) < 3: ks = [k for k in RUNS if k.endswith("1seed") and k.startswith("stgcn")]
    lab = [k.replace("stgcn_", "").replace("_1seed", "").replace("_3seed", "") for k in ks]
    try: ax[2].boxplot([E[E.model == k].auroc for k in ks], tick_labels=lab)
    except TypeError: ax[2].boxplot([E[E.model == k].auroc for k in ks], labels=lab)
    ax[2].axhline(0.5, color="#999", lw=1); ax[2].set(ylabel="AUROC per pasien", title="Skema normalisasi")
    plt.tight_layout(); plt.savefig(OUT_AN/"eda_w3.png", dpi=130); plt.show()
print("\nBerkas keluaran di:", OUT_AN)
