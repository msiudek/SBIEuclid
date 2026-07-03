"""
Photometry-system diagnostics: templfit vs MER-total vs the noise models.

Data: matched_andrea_cosmos2020.fits (403k; templfit fluxes+errs, pipeline z/logM,
COSMOS2020 LePhare z/mass) joined to phot_andrea-result.fits (2fwhm apertures +
flux_detection_total) by object_id -> both photometry systems for every galaxy.

Figures (sbi-logs/photsys/):
  ps1_sigma_vs_mag.png : per band, OBSERVED sigma(mag) in the templfit system vs
                         the total system, overlaid with the noise MODELS:
                         north_templfit (Apr/main), north_templfitJW (v3-era gen),
                         north_total (2fwhm-based).
  ps2_mag_vs_z.png     : per band, median mag vs z for templfit vs total
                         (+ their difference).
  ps3_grids.png        : (z, logM) grids for VIS and H: median templfit mag,
                         mag(total)-mag(templfit), sigma_templfit, sigma ratio.
"""
from pathlib import Path
import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "sbi-logs" / "photsys"
OUT.mkdir(parents=True, exist_ok=True)
AND = Path("/home/msiudek/myspace/projects/EUCLID/DR1/Andrea")

# SBI band order (filters_to_use.dat)
STEMS = ["h", "j", "y", "vis", "g_ext_hsc", "z_ext_hsc",
         "g_ext_decam", "r_ext_decam", "i_ext_decam", "z_ext_decam"]
NAMES = ["NISP-H", "NISP-J", "NISP-Y", "VIS", "HSC-g", "HSC-z",
         "DEC-g", "DEC-r", "DEC-i", "DEC-z"]
SNR_DET = 3.0
LN10_F = 1.0857  # sigma_mag = 1.0857 * ferr/f


def tf_cols(stem):
    if stem == "vis":
        return "flux_vis_psf", "fluxerr_vis_psf"
    return f"flux_{stem}_templfit", f"fluxerr_{stem}_templfit"


def load_joined():
    m = fits.open(AND / "matched_andrea_cosmos2020.fits")[1].data
    p = fits.open(AND / "phot_andrea-result.fits")[1].data
    pid = np.asarray(p["object_id"]).astype(np.int64)
    prow = {int(i): k for k, i in enumerate(pid)}
    moid = np.asarray(m["object_id"]).astype(np.int64)
    idx = np.array([prow.get(int(o), -1) for o in moid])
    ok = idx >= 0
    m = m[ok]; idx = idx[ok]
    print(f"joined {ok.sum()} / {len(moid)} galaxies")

    det = np.asarray(p["flux_detection_total"], float)[idx]
    vis2 = np.asarray(p["flux_vis_2fwhm_aper"], float)[idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = det / vis2

    out = {"z": np.asarray(m["cos20_lp_zbest"], float),
           "logM": np.asarray(m["cos20_lp_mass_med"], float),
           "pipez": np.asarray(m["pipeline_z"], float)}
    for st in STEMS:
        fc, ec = tf_cols(st)
        out[f"tf_f_{st}"] = np.asarray(m[fc], float)
        out[f"tf_e_{st}"] = np.asarray(m[ec], float)
        out[f"to_f_{st}"] = np.asarray(p[f"flux_{st}_2fwhm_aper"], float)[idx] * s
        out[f"to_e_{st}"] = np.asarray(p[f"fluxerr_{st}_2fwhm_aper"], float)[idx] * s
    return out


def mag_sig(f, e):
    ok = np.isfinite(f) & (f > 0) & np.isfinite(e) & (e > 0) & (f / e >= SNR_DET)
    mag = np.where(ok, -2.5 * np.log10(np.where(ok, f, 1) / 3631e6), np.nan)
    sig = np.where(ok, LN10_F * e / np.where(ok, f, 1), np.nan)
    return mag, sig


def med_prof(x, y, bins):
    c = 0.5 * (bins[:-1] + bins[1:])
    out = np.full(len(c), np.nan)
    for i in range(len(c)):
        mm = np.isfinite(x) & np.isfinite(y) & (x >= bins[i]) & (x < bins[i + 1])
        if mm.sum() > 30:
            out[i] = np.median(y[mm])
    return c, out


def centers(edges):
    e = edges
    c = [e[0] - (e[1] - e[0]) / 2]
    for j in range(len(e) - 1):
        c.append((e[j] + e[j + 1]) / 2)
    c.append(e[-1] + (e[-1] - e[-2]) / 2)
    return np.array(c)


def load_noise(pref):
    try:
        ms = np.load(ROOT / f"obs/obs_properties/mean_sigma_{pref}.npy")
        pc = np.load(ROOT / f"obs/obs_properties/percentiles_{pref}.npy")
        return ms, pc
    except FileNotFoundError:
        return None, None


def main():
    d = load_joined()

    # ---------------- ps1: sigma vs mag ----------------
    models = [("north_templfit", "Apr/main model", "tab:blue"),
              ("north_templfitJW", "v3-era model", "tab:green"),
              ("north_total", "2fwhm-total model", "tab:purple")]
    fig, ax = plt.subplots(2, 5, figsize=(22, 8), sharey=True)
    ax = ax.ravel()
    mb = np.arange(18, 27.5, 0.5)
    for i, (st, name) in enumerate(zip(STEMS, NAMES)):
        mt, sgt = mag_sig(d[f"tf_f_{st}"], d[f"tf_e_{st}"])
        mo, sgo = mag_sig(d[f"to_f_{st}"], d[f"to_e_{st}"])
        c, p1 = med_prof(mt, sgt, mb)
        _, p2 = med_prof(mo, sgo, mb)
        ax[i].plot(c, p1, "k-", lw=2.5, label="OBS templfit")
        ax[i].plot(c, p2, "r--", lw=2.5, label="OBS total(2fwhm)")
        for pref, lab, col in models:
            ms, pcs = load_noise(pref)
            if ms is None:
                continue
            ax[i].plot(centers(pcs[:, i]), ms[i], "o:", ms=3, color=col, label=f"model {lab}", alpha=0.8)
        ax[i].set_title(name, fontsize=10)
        ax[i].set_xlabel("mag"); ax[i].set_ylim(0, 0.45)
        if i == 0:
            ax[i].set_ylabel("sigma [mag]"); ax[i].legend(fontsize=6)
    fig.suptitle("sigma vs mag: OBSERVED (templfit vs total) + the 3 noise models", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "ps1_sigma_vs_mag.png", dpi=110)
    print("wrote ps1_sigma_vs_mag.png")

    # ---------------- ps2: mag vs z ----------------
    zb = np.arange(0, 5.25, 0.25)
    fig, ax = plt.subplots(2, 5, figsize=(22, 8), sharey=False)
    ax = ax.ravel()
    z = d["z"]
    for i, (st, name) in enumerate(zip(STEMS, NAMES)):
        mt, _ = mag_sig(d[f"tf_f_{st}"], d[f"tf_e_{st}"])
        mo, _ = mag_sig(d[f"to_f_{st}"], d[f"to_e_{st}"])
        c, p1 = med_prof(z, mt, zb)
        _, p2 = med_prof(z, mo, zb)
        ax[i].plot(c, p1, "k-", lw=2, label="templfit")
        ax[i].plot(c, p2, "r--", lw=2, label="total")
        a2 = ax[i].twinx()
        a2.plot(c, p2 - p1, color="tab:orange", lw=1, alpha=0.8)
        a2.axhline(0, color="tab:orange", lw=0.4, ls=":")
        a2.set_ylim(-1, 1)
        a2.tick_params(axis="y", labelcolor="tab:orange", labelsize=7)
        ax[i].invert_yaxis()
        ax[i].set_title(name, fontsize=10); ax[i].set_xlabel("z (COSMOS2020)")
        if i == 0:
            ax[i].set_ylabel("median mag (detected)"); ax[i].legend(fontsize=7)
        if i == 4:
            a2.set_ylabel("total − templfit [mag]", color="tab:orange", fontsize=8)
    fig.suptitle("median magnitude vs z: templfit (black) vs total (red); orange = difference", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "ps2_mag_vs_z.png", dpi=110)
    print("wrote ps2_mag_vs_z.png")

    # ---------------- ps3: (z, logM) grids for VIS & H ----------------
    zg = np.linspace(0, 4, 17)
    mg = np.linspace(7.5, 12, 19)
    logM = d["logM"]

    def grid(vals):
        H = np.full((len(mg) - 1, len(zg) - 1), np.nan)
        for a in range(len(mg) - 1):
            for b in range(len(zg) - 1):
                mm = (np.isfinite(vals) & np.isfinite(z) & np.isfinite(logM)
                      & (z >= zg[b]) & (z < zg[b + 1])
                      & (logM >= mg[a]) & (logM < mg[a + 1]))
                if mm.sum() > 15:
                    H[a, b] = np.median(vals[mm])
        return H

    rows = []
    for st, name in [("vis", "VIS"), ("h", "NISP-H")]:
        mt, sgt = mag_sig(d[f"tf_f_{st}"], d[f"tf_e_{st}"])
        mo, sgo = mag_sig(d[f"to_f_{st}"], d[f"to_e_{st}"])
        with np.errstate(divide="ignore", invalid="ignore"):
            rows.append((name, grid(mt), grid(mo - mt), grid(sgt), grid(sgo / sgt)))

    fig, ax = plt.subplots(2, 4, figsize=(21, 9))
    ext = [zg[0], zg[-1], mg[0], mg[-1]]
    for r, (name, g_mag, g_dmag, g_sig, g_rsig) in enumerate(rows):
        specs = [(g_mag, f"{name}: median mag (templfit)", "viridis", None, None),
                 (g_dmag, f"{name}: mag total − templfit", "coolwarm", -0.6, 0.6),
                 (g_sig, f"{name}: median sigma (templfit)", "magma", 0, 0.3),
                 (g_rsig, f"{name}: sigma total/templfit", "coolwarm", 0, 2)]
        for cidx, (G, title, cmap, vmin, vmax) in enumerate(specs):
            im = ax[r, cidx].imshow(G, origin="lower", aspect="auto", extent=ext,
                                    cmap=cmap, vmin=vmin, vmax=vmax)
            ax[r, cidx].set_title(title, fontsize=9)
            ax[r, cidx].set_xlabel("z"); ax[r, cidx].set_ylabel("logM* (COSMOS2020)")
            plt.colorbar(im, ax=ax[r, cidx], fraction=0.045)
    fig.suptitle("(z, logM) grids: magnitude & sigma, templfit vs total", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "ps3_grids.png", dpi=110)
    print("wrote ps3_grids.png")

    # console summary: system offsets per band (detected both)
    print("\nmedian(total - templfit) [mag], SNR>=3 in both:")
    for st, name in zip(STEMS, NAMES):
        mt, _ = mag_sig(d[f"tf_f_{st}"], d[f"tf_e_{st}"])
        mo, _ = mag_sig(d[f"to_f_{st}"], d[f"to_e_{st}"])
        mm = np.isfinite(mt) & np.isfinite(mo)
        print(f"  {name:7s} {np.median((mo-mt)[mm]):+.3f}  (N={mm.sum()})")


if __name__ == "__main__":
    main()
