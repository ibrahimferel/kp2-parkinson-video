import os, time, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from fogbench import splits, report, data as D
from fogbench.threshold import pick_threshold

os.environ["FOG_STGCN_SKEMA"] = "pinggul"
import models.stgcn as M

RUN_ACUAN = "stgcn_pinggul_1seed"
FS_EVAL = 32
OUT_CEK = HASIL / "analisis" / "cek_ambang"; OUT_CEK.mkdir(parents=True, exist_ok=True)

tersimpan = report.load_run(HASIL / RUN_ACUAN)
thr_simpan = {f["test_subject"]: f["threshold"] for f in tersimpan}
def punya_dua_kelas(s):
    y = np.concatenate([np.asarray(r.y) for r in recs if r.subject == s]); return y.min() != y.max()
kandidat = {s: t for s, t in thr_simpan.items() if punya_dua_kelas(s)}   # pasien uji tanpa FoG tidak informatif
tinggi = max(kandidat, key=kandidat.get); rendah = min(kandidat, key=kandidat.get)
pilihan = [("ambang tertinggi", tinggi), ("ambang terendah", rendah)]
print(f"Fold dipilih: ambang tertinggi = pasien {tinggi} ({thr_simpan[tinggi]:.2f}) | terendah = pasien {rendah} ({thr_simpan[rendah]:.2f})")

folds = splits.loso_folds({r.subject for r in recs})
Q = [5, 25, 50, 75, 95]


def prediksi(model, rek_list):
    p = np.concatenate([D.to_eval_rate(model.predict_proba(r), r.fs, FS_EVAL) for r in rek_list])
    y = np.concatenate([D.to_eval_rate(r.y, r.fs, FS_EVAL, is_label=True) for r in rek_list]).astype(int)
    n = min(len(p), len(y)); return y[:n], p[:n]


def sens_spes(y, p, t):
    pred = p >= t
    sen = (pred & (y == 1)).sum() / (y == 1).sum() if (y == 1).any() else np.nan
    spe = (~pred & (y == 0)).sum() / (y == 0).sum() if (y == 0).any() else np.nan
    return sen, spe


baris, fig_data = [], {}
for label, subj in pilihan:
    fold = next(f for f in folds if f.test_subject == subj)
    train, test = splits.split_recordings(recs, fold)
    model = M.factory(seed=0)
    t0 = time.time(); model.fit(train); print(f"[pasien uji {subj}] fit {time.time() - t0:.0f} s")
    y_tr, p_tr = prediksi(model, train); y_te, p_te = prediksi(model, test)
    thr = pick_threshold(y_tr, p_tr)
    s_tr = sens_spes(y_tr, p_tr, thr); s_te = sens_spes(y_te, p_te, thr)
    ada_dua = len(np.unique(y_te)) == 2
    baris.append({
        "fold": label, "pasien_uji": subj, "ambang_tersimpan": round(thr_simpan[subj], 2), "ambang_ulang": round(thr, 2),
        "AUROC_latih": round(roc_auc_score(y_tr, p_tr), 3),
        "AUROC_uji": round(roc_auc_score(y_te, p_te), 3) if ada_dua else np.nan,
        "FoG latih: P(p>=0,99)": round((p_tr[y_tr == 1] >= 0.99).mean(), 3),
        "FoG latih: P(p>=0,90)": round((p_tr[y_tr == 1] >= 0.90).mean(), 3),
        "bukanFoG latih: P(p<=0,01)": round((p_tr[y_tr == 0] <= 0.01).mean(), 3),
        "bukanFoG latih: P(p<=0,10)": round((p_tr[y_tr == 0] <= 0.10).mean(), 3),
        "FoG uji: P(p>=0,90)": round((p_te[y_te == 1] >= 0.90).mean(), 3) if (y_te == 1).any() else np.nan,
        "sens_latih@ambang": round(s_tr[0], 3), "spes_latih@ambang": round(s_tr[1], 3),
        "sens_uji@ambang": round(s_te[0], 3), "spes_uji@ambang": round(s_te[1], 3),
        "sens_uji@0,5": round(sens_spes(y_te, p_te, 0.5)[0], 3), "spes_uji@0,5": round(sens_spes(y_te, p_te, 0.5)[1], 3)})
    print(f"  persentil {Q} p | FoG latih {np.percentile(p_tr[y_tr == 1], Q).round(3)} | bukan-FoG latih {np.percentile(p_tr[y_tr == 0], Q).round(3)}")
    if (y_te == 1).any():
        print(f"  persentil {Q} p | FoG uji   {np.percentile(p_te[y_te == 1], Q).round(3)} | bukan-FoG uji   {np.percentile(p_te[y_te == 0], Q).round(3)}")
    u, c = np.unique(p_tr.round(3), return_counts=True); top = np.argsort(-c)[:3]
    print("  nilai p paling sering (latih): " + " | ".join(
        f"{u[i]:.3f}: {c[i] / len(p_tr):.1%} sampel, {y_tr[p_tr.round(3) == u[i]].mean():.0%} FoG" for i in top))
    fig_data[subj] = (label, y_tr, p_tr, y_te, p_te, thr)

T = pd.DataFrame(baris); T.to_csv(OUT_CEK / "cek_ambang.csv", index=False)
print("\n" + T.T.to_string(header=False))

fig, ax = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
for r, (subj, (label, y_tr, p_tr, y_te, p_te, thr)) in enumerate(fig_data.items()):
    for c, (nm, y, p) in enumerate([("pasien LATIH", y_tr, p_tr), (f"pasien UJI ({subj})", y_te, p_te)]):
        a = ax[r, c]; b = np.linspace(0, 1, 41)
        if (y == 0).any(): a.hist(p[y == 0], bins=b, density=True, alpha=.6, label="bukan-FoG")
        if (y == 1).any(): a.hist(p[y == 1], bins=b, density=True, alpha=.6, label="FoG")
        a.axvline(thr, color="k", ls="--", lw=1); a.set_title(f"{label}: {nm} | ambang {thr:.2f}", fontsize=9)
        a.set_yscale("log"); a.legend(fontsize=7)
for a in ax[1]: a.set_xlabel("probabilitas FoG")
plt.tight_layout(); plt.savefig(OUT_CEK / "cek_ambang.png", dpi=130); plt.show()
print(f"Berkas: {OUT_CEK}")
