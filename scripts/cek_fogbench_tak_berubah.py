"""Periksa bahwa fogbench/ dan config/protocol.yaml sama dengan starter Bu Bunga (aturan 1 README starter).

Pemakaian:  python scripts/cek_fogbench_tak_berubah.py
Keluar dengan kode 1 bila ada berkas yang berbeda.
"""
import hashlib, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
rujukan = [l.split(None, 1) for l in (ROOT / "docs" / "kp2" / "fogbench.sha256").read_text(encoding="utf-8").splitlines()
           if l.strip() and not l.startswith("#")]
buruk = []
for h, rel in rujukan:
    p = ROOT / rel.strip()
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != h:
        buruk.append(rel.strip())
print("Semua berkas sama dengan starter." if not buruk else "BERBEDA/HILANG: " + ", ".join(buruk))
sys.exit(1 if buruk else 0)
