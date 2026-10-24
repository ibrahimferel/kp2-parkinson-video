# Sel L: sensitivitas cut-off frac_valid pada uji lag (CPU, ~2-3 menit; tidak butuh GPU)
import os
import numpy as np, pandas as pd
from pathlib import Path
BASE = Path(os.environ.get("KP_BASE", "/content/drive/MyDrive/KP-TA MCI"))
FS_TARGET = 32
RES = BASE / "05. Week 2" / "resampled_32hz"
OUT_L = BASE / "06. Week 3" / "hasil" / "analisis"; OUT_L.mkdir(parents=True, exist_ok=True)

def lag_rekaman(df, maks=96):
    """Salinan Cell 6 Week 2: lag (detik) yang memaksimalkan korelasi kecepatan pergelangan kaki vs besar gyro."""
    a = df[["xy_54", "xy_55", "xy_56", "xy_57"]].to_numpy()
    dL = np.hypot(np.diff(a[:, 0]), np.diff(a[:, 1])); dR = np.hypot(np.diff(a[:, 2]), np.diff(a[:, 3]))
    with np.errstate(all="ignore"):
        spd = np.nanmean(np.vstack([dL, dR]), axis=0)
    gyr = np.sqrt((df[["gyr_ml", "gyr_ap", "gyr_si"]].to_numpy() ** 2).sum(1))[1:]
    env = lambda x: pd.Series(x).rolling(32, min_periods=8, center=True).mean().to_numpy()
    zs = lambda x: np.nan_to_num((x - np.nanmean(x)) / (np.nanstd(x) + 1e-8), nan=0.0)
    p, g = zs(env(spd)), zs(env(gyr)); kor = {}
    for k in range(-maks, maks + 1):
        x, y = (p[k:], g[:len(g) - k]) if k >= 0 else (p[:k], g[-k:])
        kor[k] = np.mean(x * y)
    kb = max(kor, key=kor.get)
    return kb / FS_TARGET, kor[kb]

baris = []
for f in sorted(RES.glob("PDFE*.parquet")):
    df = pd.read_parquet(f)
    fv = float(df["valid_ankles"].mean())
    lag, kor = lag_rekaman(df)
    baris.append((f.stem, fv, lag, kor))
L = pd.DataFrame(baris, columns=["rekaman", "frac_valid", "lag_detik", "korelasi_maks"])
L.to_csv(OUT_L / "lag_semua_rekaman.csv", index=False)

hasil = []
for cut in (0.5, 0.6, 0.7, 0.8):
    s = L[L.frac_valid >= cut]
    hasil.append(dict(cutoff=cut, n=len(s), median_lag=s.lag_detik.median(),
                      q25=s.lag_detik.quantile(.25), q75=s.lag_detik.quantile(.75),
                      lebar_IQR=s.lag_detik.quantile(.75) - s.lag_detik.quantile(.25),
                      n_korelasi_lemah_lt_0_2=int((s.korelasi_maks < 0.2).sum())))
H_ = pd.DataFrame(hasil).round(2); H_.to_csv(OUT_L / "sensitivitas_cutoff_lag.csv", index=False)
print(H_.to_string(index=False))
print("\nCek: pada cutoff 0,7 harapan n = 24, median +0,44 s (Week 2). n semua rekaman:", len(L))
