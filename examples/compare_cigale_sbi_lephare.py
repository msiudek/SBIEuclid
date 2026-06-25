"""
3-way comparison for the Euclid-bands "bands vs method" test:
CIGALE(Euclid bands) vs SBI(Euclid bands) vs full-band LePhare, same galaxies.
Decomposes the SBI bias into a band-coverage floor (CIGALE-LePhare) and an
SBI-specific excess (SBI-CIGALE).
"""
from pathlib import Path
import numpy as np, warnings
warnings.filterwarnings("ignore")
from astropy.table import Table, join
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path("sbi-logs/cigale_euclidbands")
res = Table.read(D / "results.fits")
ref = Table.read(D / "cigale_euclid_reference.csv")
res.keep_columns(["id", "bayes.stellar.m_star", "best.reduced_chi_square"])
m = join(ref, res, keys="id")

logM_cig = np.log10(np.array(m["bayes.stellar.m_star"]))
logM_sbi = np.array(m["logM_sbi"]); logM_lp = np.array(m["logM_lephare"])
z = np.array(m["redshift"]); chi2 = np.array(m["best.reduced_chi_square"])
ok = np.isfinite(logM_cig) & np.isfinite(logM_lp) & np.isfinite(logM_sbi) & (chi2 < 20)

def nmad(x): x = x[np.isfinite(x)]; return 1.4826 * np.median(np.abs(x - np.median(x)))
bc, bs = logM_cig - logM_lp, logM_sbi - logM_lp
ZB = [(0, 0.5), (0.5, 1), (1, 2), (2, 3), (3, 5)]
zc = [0.5 * (a + b) for a, b in ZB]
mc = [np.median(bc[ok & (z >= a) & (z < b)]) for a, b in ZB]
ms = [np.median(bs[ok & (z >= a) & (z < b)]) for a, b in ZB]
gap = [s - c for s, c in zip(ms, mc)]

fig, ax = plt.subplots(1, 3, figsize=(17, 5))
ax[0].plot(zc, mc, "o-", color="C0", label="CIGALE − LePhare (band floor)")
ax[0].plot(zc, ms, "s-", color="C3", label="SBI − LePhare")
ax[0].plot(zc, gap, "^--", color="C2", label="SBI − CIGALE (SBI-specific)")
ax[0].axhline(0, color="k", lw=0.8)
ax[0].set(xlabel="z", ylabel="median Δ logM", title="Euclid-bands bias decomposition")
ax[0].legend(fontsize=9)
lim = [6, 12]
for a, (mm, ttl) in zip(ax[1:], [(logM_cig, "CIGALE"), (logM_sbi, "SBI")]):
    sc = a.scatter(logM_lp[ok], mm[ok], c=z[ok], s=12, cmap="viridis", vmin=0, vmax=4)
    a.plot(lim, lim, "k--", lw=1)
    b = mm[ok] - logM_lp[ok]
    a.set(xlim=lim, ylim=lim, xlabel="logM LePhare (full band)", ylabel=f"logM {ttl} (Euclid bands)",
          title=f"{ttl}: bias {np.median(b):+.2f}, NMAD {nmad(b):.2f}")
    plt.colorbar(sc, ax=a, label="z")
fig.tight_layout()
fig.savefig(D / "cigale_sbi_lephare.png", dpi=120)
print(f"N good={ok.sum()}")
print(f"band floor  (CIGALE-LP): {np.median(bc[ok]):+.3f}")
print(f"SBI total   (SBI-LP):    {np.median(bs[ok]):+.3f}")
print(f"SBI-specific(SBI-CIG):   {np.median((logM_sbi-logM_cig)[ok]):+.3f}")
print("gap vs z:", [f"{g:+.2f}" for g in gap])
print(f"wrote {D/'cigale_sbi_lephare.png'}")
