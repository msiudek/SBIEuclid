"""
Color comparison: atlas vs COSMOS-Web in (logM, z) cells.

Compares SED colors (flux ratios between bands) between the JWST atlas
and real COSMOS-Web observations. A color mismatch at fixed (logM, z)
means the atlas SEDs have the wrong SED shape — independent of mass
normalization.

Colors compared:
  - F115W − F150W  (UV/blue slope)
  - F150W − F277W  (optical/NIR break)
  - F277W − F444W  (NIR slope)
  - F115W − F444W  (full baseline)

Outputs:
  color_grid_<color>.png   — 2D (logM, z) grid: real / atlas / delta
  color_distribution.png   — histograms per z bin
  color_offset_summary.png — median offset per color band as function of z and logM

Usage:
    python examples/color_comparison_jwst.py \
        --atlas-name atlas_jwst_50000_Nparam_2.dbatlas \
        --outdir sbi-logs/color_comparison_jwst
"""

import argparse
from pathlib import Path

import hickle
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

ROOT    = Path(__file__).resolve().parents[1]
LIB_DIR = ROOT / "library"
CATALOG = ROOT / "obs" / "obs_properties" / "COSMOS" / "COSMOSWeb_mastercatalog_v1.fits"

FILTER_STEMS = ["f115w", "f150w", "f277w", "f444w"]
FILTER_NAMES = ["F115W", "F150W", "F277W", "F444W"]
LAM_EFF_UM   = np.array([1.154, 1.501, 2.762, 4.408])
N_FILT = 4

Z_EDGES = np.arange(0.0, 5.5, 0.5)
M_EDGES = np.arange(6.0, 13.0, 0.5)

# Color pairs: (band_a_idx, band_b_idx, label)  — color = mag_a - mag_b
COLOR_PAIRS = [
    (0, 1, "F115W−F150W"),
    (1, 2, "F150W−F277W"),
    (2, 3, "F277W−F444W"),
    (0, 3, "F115W−F444W"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--atlas-name", default="atlas_jwst_50000_Nparam_2.dbatlas")
    p.add_argument("--outdir",     default="sbi-logs/color_comparison_jwst")
    p.add_argument("--snr-min",    type=float, default=3.0,
                   help="Minimum SNR per band for real galaxies (default: 3)")
    return p.parse_args()


def flux_to_mag(flux_ujy):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(flux_ujy > 0, -2.5 * np.log10(flux_ujy / 3631e6), np.nan)


def load_atlas(atlas_name):
    data  = hickle.load(str(LIB_DIR / atlas_name))
    logM  = np.array(data["mstar"], dtype=float)
    z     = np.array(data["zval"],  dtype=float)
    seds  = np.array(data["sed"],   dtype=float)   # (N, 4) µJy
    logSFR = np.array(data["sfr"],  dtype=float)
    return logM, z, seds, logSFR


def load_cosmos_web(snr_min=3.0):
    """Load COSMOS-Web photometry, SNR >= snr_min in all 4 bands."""
    print("Loading COSMOS-Web catalog...")
    phot = Table.read(str(CATALOG), hdu=1)
    ref  = Table.read(str(CATALOG), hdu=2)

    fluxes   = np.zeros((len(phot), N_FILT), dtype=float)
    fluxerrs = np.zeros((len(phot), N_FILT), dtype=float)
    for j, stem in enumerate(FILTER_STEMS):
        f = np.array(phot[f"flux_aper_{stem}"], dtype=float)
        e = np.array(phot[f"flux_err_aper_{stem}"], dtype=float)
        fluxes[:, j]   = f[:, 0] if f.ndim == 2 else f
        fluxerrs[:, j] = e[:, 0] if e.ndim == 2 else e

    # SNR cut: all 4 bands must have SNR >= snr_min
    with np.errstate(divide='ignore', invalid='ignore'):
        snr = np.abs(fluxes / np.where(fluxerrs > 0, fluxerrs, np.nan))
    good = np.all(snr >= snr_min, axis=1)

    # Reference z and logM
    z_col = next((c for c in ["zpdf_med", "z_lephare", "z"] if c in ref.colnames), None)
    m_col = next((c for c in ["mass_med", "logM_lephare", "logM"] if c in ref.colnames), None)
    z_ref = np.array(ref[z_col], dtype=float)
    m_ref = np.array(ref[m_col], dtype=float)

    good &= np.isfinite(z_ref) & (z_ref > 0) & np.isfinite(m_ref) & (m_ref > 4)

    print(f"  {good.sum()} galaxies with SNR≥{snr_min} in all 4 bands (of {len(phot)})")
    return fluxes[good], fluxerrs[good], z_ref[good], m_ref[good]


def cell_median(values, z_arr, logM_arr, min_count=5):
    nz = len(Z_EDGES) - 1
    nm = len(M_EDGES) - 1
    grid  = np.full((nm, nz), np.nan)
    count = np.zeros((nm, nz), dtype=int)
    for iz in range(nz):
        for im in range(nm):
            mask = (
                (z_arr    >= Z_EDGES[iz]) & (z_arr    < Z_EDGES[iz+1]) &
                (logM_arr >= M_EDGES[im]) & (logM_arr < M_EDGES[im+1])
            )
            if mask.sum() >= min_count:
                grid[im, iz]  = np.nanmedian(values[mask])
                count[im, iz] = mask.sum()
    return grid, count


def main():
    args   = parse_args()
    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    atlas_logM, atlas_z, atlas_seds, atlas_logSFR = load_atlas(args.atlas_name)
    flux_real, fluxerr_real, z_real, logM_real = load_cosmos_web(args.snr_min)

    # Convert to magnitudes
    mag_real  = flux_to_mag(flux_real)     # (N_real, 4)
    mag_atlas = flux_to_mag(atlas_seds)    # (N_atlas, 4)

    # ──────────────────────────────────────────────────────────────────────
    # 1. Color grids: median color per (logM, z) cell
    # ──────────────────────────────────────────────────────────────────────
    print("\nComputing color grids...")

    for ia, ib, clabel in COLOR_PAIRS:
        color_real  = mag_real[:, ia]  - mag_real[:, ib]
        color_atlas = mag_atlas[:, ia] - mag_atlas[:, ib]

        grid_real,  cnt_real  = cell_median(color_real,  z_real,   logM_real)
        grid_atlas, cnt_atlas = cell_median(color_atlas, atlas_z,  atlas_logM)
        grid_delta = grid_atlas - grid_real   # positive = atlas bluer (more short-λ flux)

        # Mask cells with too few galaxies
        grid_real[cnt_real   < 5]  = np.nan
        grid_atlas[cnt_atlas < 5]  = np.nan

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Color: {clabel}  (mag_a − mag_b, positive = bluer)\n"
                     f"Δcolor > 0 → atlas too blue | Δcolor < 0 → atlas too red", fontsize=11)

        vmax = max(np.nanpercentile(np.abs(grid_real[np.isfinite(grid_real)]), 95),
                   np.nanpercentile(np.abs(grid_atlas[np.isfinite(grid_atlas)]), 95))
        vmin_c = np.nanmin([np.nanpercentile(grid_real[np.isfinite(grid_real)], 5),
                             np.nanpercentile(grid_atlas[np.isfinite(grid_atlas)], 5)])
        vmax_c = np.nanmax([np.nanpercentile(grid_real[np.isfinite(grid_real)], 95),
                             np.nanpercentile(grid_atlas[np.isfinite(grid_atlas)], 95)])

        ext = [Z_EDGES[0], Z_EDGES[-1], M_EDGES[0], M_EDGES[-1]]

        im0 = axes[0].imshow(grid_real,  origin="lower", cmap="RdYlBu_r", aspect="auto",
                              extent=ext, vmin=vmin_c, vmax=vmax_c)
        axes[0].set_title(f"COSMOS-Web (real)\nN cells={np.isfinite(grid_real).sum()}")
        axes[0].set_xlabel("z"); axes[0].set_ylabel("logM*")
        plt.colorbar(im0, ax=axes[0], label=f"{clabel} [mag]")

        im1 = axes[1].imshow(grid_atlas, origin="lower", cmap="RdYlBu_r", aspect="auto",
                              extent=ext, vmin=vmin_c, vmax=vmax_c)
        axes[1].set_title(f"Atlas (FSPS mock)\nN cells={np.isfinite(grid_atlas).sum()}")
        axes[1].set_xlabel("z")
        plt.colorbar(im1, ax=axes[1], label=f"{clabel} [mag]")

        vlim = min(2.0, np.nanpercentile(np.abs(grid_delta[np.isfinite(grid_delta)]), 90))
        im2 = axes[2].imshow(grid_delta, origin="lower", cmap="RdBu_r", aspect="auto",
                              extent=ext, vmin=-vlim, vmax=vlim)
        axes[2].set_title(f"Δcolor = atlas − real\n(+) atlas too blue  (−) atlas too red")
        axes[2].set_xlabel("z")
        plt.colorbar(im2, ax=axes[2], label="Δcolor [mag]")

        plt.tight_layout()
        label_safe = clabel.replace("−", "m").replace(" ", "_")
        out = outdir / f"color_grid_{label_safe}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  ✓ {out.name}  |  median Δ={np.nanmedian(grid_delta[np.isfinite(grid_delta)]):+.3f} mag")

    # ──────────────────────────────────────────────────────────────────────
    # 2. Color distributions per redshift bin
    # ──────────────────────────────────────────────────────────────────────
    print("\nPlotting color distributions per z bin...")
    zbins   = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0)]
    zcolors = ["steelblue", "seagreen", "darkorange", "crimson"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Color distributions: COSMOS-Web (solid) vs atlas (dashed) per z bin", fontsize=12)

    for ax, (ia, ib, clabel) in zip(axes.flat, COLOR_PAIRS):
        color_real  = mag_real[:, ia]  - mag_real[:, ib]
        color_atlas = mag_atlas[:, ia] - mag_atlas[:, ib]

        for (zlo, zhi), zc in zip(zbins, zcolors):
            mr = (z_real  >= zlo) & (z_real  < zhi) & np.isfinite(color_real)
            ma = (atlas_z >= zlo) & (atlas_z < zhi) & np.isfinite(color_atlas)
            if mr.sum() < 5 or ma.sum() < 5:
                continue

            bins = np.linspace(-2, 4, 50)
            ax.hist(color_real[mr],  bins=bins, density=True, histtype="step",
                    color=zc, lw=2,   label=f"z=[{zlo},{zhi}) real")
            ax.hist(color_atlas[ma], bins=bins, density=True, histtype="step",
                    color=zc, lw=1.5, linestyle="--", label=f"z=[{zlo},{zhi}) atlas")

        ax.set_xlabel(f"{clabel} [mag]"); ax.set_ylabel("Density")
        ax.set_title(clabel)
        ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    out = outdir / "color_distributions.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"✓ Saved: {out.name}")

    # ──────────────────────────────────────────────────────────────────────
    # 3. Summary: median Δcolor vs z (all logM combined)
    # ──────────────────────────────────────────────────────────────────────
    print("\nPlotting Δcolor vs z summary...")
    z_centers = 0.5 * (Z_EDGES[:-1] + Z_EDGES[1:])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(0, color='k', lw=1, ls='--')
    styles = ['-o', '-s', '-^', '-D']

    for (ia, ib, clabel), sty in zip(COLOR_PAIRS, styles):
        color_real  = mag_real[:, ia]  - mag_real[:, ib]
        color_atlas = mag_atlas[:, ia] - mag_atlas[:, ib]
        delta_z = []
        for iz in range(len(Z_EDGES) - 1):
            mr = (z_real  >= Z_EDGES[iz]) & (z_real  < Z_EDGES[iz+1]) & np.isfinite(color_real)
            ma = (atlas_z >= Z_EDGES[iz]) & (atlas_z < Z_EDGES[iz+1]) & np.isfinite(color_atlas)
            if mr.sum() >= 5 and ma.sum() >= 5:
                delta_z.append(np.nanmedian(color_atlas[ma]) - np.nanmedian(color_real[mr]))
            else:
                delta_z.append(np.nan)
        ax.plot(z_centers, delta_z, sty, label=clabel, lw=2, ms=7)

    ax.set_xlabel("Redshift z", fontsize=12)
    ax.set_ylabel("Δcolor = atlas − real [mag]\n(+) atlas too blue  (−) atlas too red", fontsize=11)
    ax.set_title("Color offset between FSPS atlas and COSMOS-Web observations\n"
                 "Systematic Δcolor → wrong SED shape → explains mass bias", fontsize=11)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 5)
    ax.set_ylim(-2, 3)

    plt.tight_layout()
    out = outdir / "color_offset_vs_z.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"✓ Saved: {out.name}")

    # ──────────────────────────────────────────────────────────────────────
    # 4. Print summary table
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("COLOR OFFSET SUMMARY  (atlas − real)  [+] = atlas bluer")
    print("=" * 65)
    print(f"{'Color':<18} {'All z':>8} {'z<1':>8} {'1<z<2':>8} {'2<z<3':>8} {'z>3':>8}")
    print("-" * 65)
    zbins_summ = [(0, 99), (0, 1), (1, 2), (2, 3), (3, 9)]
    for ia, ib, clabel in COLOR_PAIRS:
        color_real  = mag_real[:, ia]  - mag_real[:, ib]
        color_atlas = mag_atlas[:, ia] - mag_atlas[:, ib]
        row = f"{clabel:<18}"
        for zlo, zhi in zbins_summ:
            mr = (z_real  >= zlo) & (z_real  < zhi) & np.isfinite(color_real)
            ma = (atlas_z >= zlo) & (atlas_z < zhi) & np.isfinite(color_atlas)
            if mr.sum() >= 5 and ma.sum() >= 5:
                d = np.nanmedian(color_atlas[ma]) - np.nanmedian(color_real[mr])
                row += f" {d:>+8.3f}"
            else:
                row += f" {'---':>8}"
        print(row)
    print("=" * 65)
    print("\nInterpretation:")
    print("  Δ(F115W−F444W) > 0 → atlas much bluer than real → FSPS SFH/dust prior too blue")
    print("  Δ(F277W−F444W) > 0 → excess short-NIR in atlas → possibly wrong AGB/dust")
    print("  z-dependent Δcolor → prior doesn't capture z-evolution of galaxy colors correctly")


if __name__ == "__main__":
    main()
