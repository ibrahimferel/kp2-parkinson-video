# Sel F: ringkasan hasil 35 fold (jalankan SETELAH Sel A)
import json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from fogbench import report
from fogbench.schema import validate_dir
from fogbench.metrics import bootstrap_ci

URUT = ["li_1seed", "li_3seed", "stgcn_pinggul_1seed", "stgcn_pinggul_3seed", "stgcn_skala_1seed", "stgcn_piksel_1seed", "stgcn_skala_3seed", "stgcn_piksel_3seed"]
RUNS = {k: HASIL/k for k in URUT if (HASIL/k).exists() and len(list((HASIL/k).glob("fold_*.json"))) >= 1}
for k, v in RUNS.items():
    n = len(list(v.glob("fold_*.json")))
    if n < 35: print(f"PERINGATAN: {k} baru {n}/35 fold -> dilewati"); 
RUNS = {k: v for k, v in RUNS.items() if len(list(v.glob("fold_*.json"))) >= 35}
SERING_DOMINAN = [1, 7, 19]          # pasien dengan >= 90% FoG (Week 2)

def muat(folder):
    return report.load_run(folder)

def per_seed_tabel(folds):
    """AUROC/AUPRC gabungan dan AUROC per pasien untuk SETIAP seed (bebas ambang)."""
    n_seed = len(folds[0]["p_per_seed"]); baris = []
    for s in range(n_seed):
        ys = np.concatenate([f["y_true"] for f in folds]); ps = np.concatenate([f["p_per_seed"][s] for f in folds])
        pas = []
        for f in folds:
            y = np.asarray(f["y_true"]);
            if y.min() != y.max(): pas.append(roc_auc_score(y, f["p_per_seed"][s]))
        baris.append(dict(seed=folds[0]["seeds"][s], auroc_gabungan=roc_auc_score(ys, ps),
                          auprc_gabungan=average_precision_score(ys, ps), auroc_per_pasien=np.mean(pas)))
    return pd.DataFrame(baris)

per_pasien, ringkas = {}, []
for nama, folder in RUNS.items():
    ok, masalah = validate_dir(folder, 35)
    folds = muat(folder)
    print(f"\n{'='*70}\n{nama} | fold: {len(folds)} | validasi: {'OK' if ok else masalah}")
    s = report.summarize(folder); p, q = s["pooled"], s["per_subject_ringkas"]
    print(f"Gabungan (p_mean): AUROC {p['auroc']:.3f} | AUPRC {p['auprc']:.3f} | sens {p['sensitivitas']:.3f} | "
          f"spes {p['spesifisitas']:.3f} | F1 {p['f1']:.3f} | F1 episode {p['episode_f1']:.3f} | prevalensi {p['prevalensi']:.3f}")
    print(f"Per pasien: n={q['n']} | AUROC {q['auroc_mean']:.3f} +/- {q['auroc_sd']:.3f} | median {q['auroc_median']:.3f} | IK95% bootstrap {s['per_subject_ringkas']['auroc_ci95'][0]:.3f}..{s['per_subject_ringkas']['auroc_ci95'][1]:.3f}")
    print(f"Waktu latih total: {s['train_time_s_total']/3600:.2f} jam | inferensi median: {s['infer_time_ms_median']:.0f} ms | parameter: {s['model_card'].get('n_parameter')}")
    print("Kartu metode memuat dropout:", "dropout" in s["model_card"].get("hiperparameter", {}))
    ps = per_seed_tabel(folds)
    print(ps.round(3).to_string(index=False))
    if len(ps) > 1:
        print("Rerata +/- SD antar seed:", {c: f"{ps[c].mean():.3f} +/- {ps[c].std(ddof=1):.3f}" for c in ps.columns if c != "seed"})
    thr = np.array([f["threshold"] for f in folds])
    print(f"Ambang per fold: median {np.median(thr):.2f} | rentang {thr.min():.2f}..{thr.max():.2f} | fold ambang > 0,9: {int((thr > 0.9).sum())}")
    # tanpa 3 pasien dominan (FoG >= 90%)
    sisa = [f for f in folds if f["test_subject"] not in SERING_DOMINAN]
    y = np.concatenate([f["y_true"] for f in sisa]); pp = np.concatenate([f["p_mean"] for f in sisa])
    print(f"AUROC gabungan tanpa pasien {SERING_DOMINAN}: {roc_auc_score(y, pp):.3f}")
    per_pasien[nama] = {m["test_subject"]: m["auroc"] for m in s["per_subject"]}
    ringkas.append(dict(model=nama, auroc_gab=p["auroc"], auprc_gab=p["auprc"], sens=p["sensitivitas"], spes=p["spesifisitas"],
                        f1=p["f1"], f1_episode=p["episode_f1"], auroc_pasien=q["auroc_mean"], sd_pasien=q["auroc_sd"], n_pasien=q["n"]))

print("\n" + "="*70 + "\nTabel resmi kerangka (report.table):")
print(report.table(list(RUNS.values())))
def uji(ref, lain, catatan):
    if ref in RUNS and all(x in RUNS for x in lain):
        print(f"\n{catatan} (referensi: {ref})"); print(report.paired_tests(RUNS[ref], [RUNS[x] for x in lain]))
uji("li_1seed", ["stgcn_pinggul_1seed"], "Li vs ST-GCN, sama-sama 1 seed")
uji("li_3seed", ["stgcn_pinggul_3seed"], "Li vs ST-GCN, sama-sama 3 seed (perbandingan utama)")
uji("stgcn_pinggul_1seed", ["stgcn_skala_1seed", "stgcn_piksel_1seed"], "RQ2: skema normalisasi, 1 seed (Holm untuk 2 pembanding)")
uji("stgcn_pinggul_3seed", ["stgcn_skala_3seed", "stgcn_piksel_3seed"], "RQ2: skema normalisasi, 3 seed (Holm untuk 2 pembanding)")

# AUROC per pasien -> CSV (untuk grafik dan lampiran)
tab = pd.DataFrame(per_pasien).sort_index(); tab.index.name = "pasien"
OUT_AN = HASIL / "analisis"; OUT_AN.mkdir(exist_ok=True)
tab.round(4).to_csv(OUT_AN / "auroc_per_pasien.csv"); pd.DataFrame(ringkas).round(4).to_csv(OUT_AN / "ringkasan_model.csv", index=False)
print("\nAUROC per pasien (26 pasien yang punya dua kelas):"); print(tab.round(3).to_string())

# kesetaraan: seed 0 pada Li 3 seed harus sama dengan Li 1 seed (Li bersifat deterministik)
if "li_1seed" in RUNS and "li_3seed" in RUNS:
    a, b = muat(RUNS["li_1seed"]), muat(RUNS["li_3seed"])
    sama = [bool(np.allclose(x["p_mean"], y["p_per_seed"][0], atol=1e-6)) for x, y in zip(a, b)]
    print(f"\nLi: fold dengan p(1 seed) == p(seed 0 dari 3 seed): {sum(sama)}/{len(sama)}")
