from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from fogbench import data as D, splits, schema
from fogbench.evaluate import protocol
from fogbench.threshold import pick_threshold


def satu_fold(model_factory, recs, subjek_uji, out_dir, model_name, dataset="figshare", seeds=None, fs_eval=None):
    """Salinan isi run_loso (fogbench/evaluate.py) untuk SATU fold; logika tidak diubah."""
    p = protocol()
    seeds = list(seeds or p["seeds"]); fs_eval = int(fs_eval or p["fs_evaluasi_hz"])
    fold = next(f for f in splits.loso_folds({r.subject for r in recs}) if f.test_subject == subjek_uji)
    train, test = splits.split_recordings(recs, fold)
    p_seeds, t_train = [], 0.0
    for sd in seeds:
        model = model_factory(seed=sd)
        t0 = time.perf_counter(); model.fit(train); t_train += time.perf_counter() - t0
        p_seeds.append(np.concatenate([D.to_eval_rate(model.predict_proba(r), r.fs, fs_eval) for r in test]))
    p_mean = np.mean(p_seeds, axis=0)
    y_eval = np.concatenate([D.to_eval_rate(r.y, r.fs, fs_eval, is_label=True) for r in test])
    n = min(len(p_mean), len(y_eval)); p_mean, y_eval = p_mean[:n], y_eval[:n]; p_seeds = [q[:n] for q in p_seeds]
    model_t = model_factory(seed=seeds[0]); model_t.fit(train)          # ambang dari pasien latih saja
    y_tr = np.concatenate([D.to_eval_rate(r.y, r.fs, fs_eval, is_label=True) for r in train])
    p_tr = np.concatenate([D.to_eval_rate(model_t.predict_proba(r), r.fs, fs_eval) for r in train])
    m = min(len(y_tr), len(p_tr)); thr = pick_threshold(y_tr[:m], p_tr[:m])
    t0 = time.perf_counter(); _ = model_t.predict_proba(test[0]); infer_ms = (time.perf_counter() - t0) * 1000
    return schema.write_fold(Path(out_dir) / f"fold_{fold.index:02d}.json", model=model_name, dataset=dataset,
                             fold=fold.index, test_subject=fold.test_subject, train_subjects=list(fold.train_subjects),
                             fs_eval=fs_eval, threshold=thr, y_true=y_eval, p_mean=p_mean, p_per_seed=p_seeds,
                             seeds=seeds, train_time_s=t_train, infer_time_ms=infer_ms, model_card=model_t.card())


def buat_jalankan_loso(recs, dataset="figshare", verbose=True):
    """Kembalikan jalankan_loso(factory, nama, out_dir, seeds) untuk `recs`; fold yang berkasnya ada dilewati."""
    def jalankan_loso(factory, nama, out_dir, seeds):
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        folds = splits.loso_folds({r.subject for r in recs})
        for f in folds:
            if (out_dir / f"fold_{f.index:02d}.json").exists():
                continue
            t0 = time.time()
            satu_fold(factory, recs, f.test_subject, out_dir, nama, dataset=dataset, seeds=seeds)
            if verbose:
                print(f"fold {f.index:02d}/{len(folds)} pasien {f.test_subject} | {time.time()-t0:.0f} s", flush=True)
    return jalankan_loso
