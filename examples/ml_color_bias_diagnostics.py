"""
M/L and color-bias diagnostics for SBI vs COSMOS-Web

- Computes M/L for both SBI and COSMOS-Web using NISP-H
- Plots M/L_SBI vs M/L_CWeb, Δlog(M/L) vs z, ΔlogM vs color, ΔlogM vs z
- Requires: selection_sweep_summary.csv, matched_euclid_cosmosweb.fits, filters_to_use.dat
"""

import numpy as np
import pandas as pd
from astropy.io import fits
from pathlib import Path
import matplotlib.pyplot as plt

from sbipix.utils.sed_utils import load_filter_metadata, mag_to_flux_ujy

# --- Config ---
LOGDIR = Path("sbi-logs/mock_as_data_sweep_v5.0")
CATALOG = Path("obs/obs_properties/COSMOS-Web/matched_euclid_cosmosweb.fits")
FILTER_META = load_filter_metadata("filters_to_use.dat", filt_dir="obs/obs_properties")


# --- Load inference results (SBI vs COSMOS-Web) ---
INFERDIR = Path("sbi-logs/inference_cosmosweb_v5.0")
inference = np.load(INFERDIR / "inference_results.npz")



# Use selection indices to extract photometry for the correct galaxies
sel = inference["selected_indices"]
logM_SBI = inference["logM_sbi"]
logM_CWeb = inference["logM_cosmosweb"]
z = inference["z"]




# --- Load COSMOS-Web catalog ---
with fits.open(CATALOG) as hdul:
    cat = hdul[1].data

# --- Band indices ---
def band_idx(band):
    for i, f in enumerate(FILTER_META):
        if f['short'].lower() == band.lower():
            return i
    raise ValueError(f"Band {band} not found")

idx_H = band_idx("NISP-H")
idx_VIS = band_idx("VIS")
idx_Y = band_idx("NISP-Y")
idx_J = band_idx("NISP-J")

# --- Extract photometry ---
import numpy as np
def flux_to_mag(flux): return -2.5 * np.log10(flux) + 23.9


mag_H = flux_to_mag(cat["flux_h_templfit"])[sel]
mag_VIS = flux_to_mag(cat["flux_vis_psf"])[sel]
mag_Y = flux_to_mag(cat["flux_y_templfit"])[sel]
mag_J = flux_to_mag(cat["flux_j_templfit"])[sel]


flux_H = mag_to_flux_ujy(mag_H)
flux_VIS = mag_to_flux_ujy(mag_VIS)
flux_Y = mag_to_flux_ujy(mag_Y)
flux_J = mag_to_flux_ujy(mag_J)


# logM_SBI, logM_CWeb, z already loaded from inference_results.npz


ml_SBI = 10**logM_SBI / flux_H
ml_CWeb = 10**logM_CWeb / flux_H


delta_logM = logM_SBI - logM_CWeb
delta_logML = np.log10(ml_SBI) - np.log10(ml_CWeb)


color_VIS_Y = mag_VIS - mag_Y
color_Y_J = mag_Y - mag_J
color_J_H = mag_J - mag_H

# --- Plots ---
plt.figure()
plt.scatter(ml_CWeb, ml_SBI, alpha=0.5)
plt.plot([np.nanmin(ml_CWeb), np.nanmax(ml_CWeb)], [np.nanmin(ml_CWeb), np.nanmax(ml_CWeb)], 'k--')
plt.xlabel("M/L (COSMOS-Web)")
plt.ylabel("M/L (SBI)")
plt.title("M/L Comparison (NISP-H)")
plt.savefig(LOGDIR / "ml_comparison.png")

plt.figure()
plt.scatter(z, delta_logML, alpha=0.5)
plt.xlabel("Redshift")
plt.ylabel(r"$\Delta \log(M/L)$ (SBI - CWeb)")
plt.title(r"$\Delta \log(M/L)$ vs z")
plt.savefig(LOGDIR / "delta_logML_vs_z.png")

plt.figure()
plt.scatter(color_VIS_Y, delta_logM, alpha=0.5, label="VIS-Y")
plt.scatter(color_Y_J, delta_logM, alpha=0.5, label="Y-J")
plt.scatter(color_J_H, delta_logM, alpha=0.5, label="J-H")
plt.xlabel("Color")
plt.ylabel(r"$\Delta \log M$")
plt.legend()
plt.title(r"$\Delta \log M$ vs Color")
plt.savefig(LOGDIR / "delta_logM_vs_color.png")

plt.figure()
plt.scatter(z, delta_logM, alpha=0.5)
plt.xlabel("Redshift")
plt.ylabel(r"$\Delta \log M$")
plt.title(r"$\Delta \log M$ vs z")
plt.savefig(LOGDIR / "delta_logM_vs_z.png")

# --- Additional plots: Δlog(M/L) vs Av, sSFR, metallicity ---
extra_keys = inference.files

if "Av" in extra_keys:
    Av = inference["Av"]
    plt.figure()
    plt.scatter(Av, delta_logML, alpha=0.5)
    plt.xlabel("A_v (SBI)")
    plt.ylabel(r"$\Delta \log(M/L)$ (SBI - CWeb)")
    plt.title(r"$\Delta \log(M/L)$ vs $A_v$")
    plt.savefig(LOGDIR / "delta_logML_vs_Av.png")

if "log_sSFR" in extra_keys:
    log_sSFR = inference["log_sSFR"]
    plt.figure()
    plt.scatter(log_sSFR, delta_logML, alpha=0.5)
    plt.xlabel("log(sSFR) (SBI)")
    plt.ylabel(r"$\Delta \log(M/L)$ (SBI - CWeb)")
    plt.title(r"$\Delta \log(M/L)$ vs log(sSFR)")
    plt.savefig(LOGDIR / "delta_logML_vs_log_sSFR.png")

if "metallicity" in extra_keys:
    metallicity = inference["metallicity"]
    plt.figure()
    plt.scatter(metallicity, delta_logML, alpha=0.5)
    plt.xlabel("[M/H] (SBI)")
    plt.ylabel(r"$\Delta \log(M/L)$ (SBI - CWeb)")
    plt.title(r"$\Delta \log(M/L)$ vs [M/H]")
    plt.savefig(LOGDIR / "delta_logML_vs_metallicity.png")

print("Plots saved to", LOGDIR)


# --- Additional plots: Δlog(M/L) vs median sSFR, Av, [M/H] from posterior samples ---
posteriors = inference["posteriors"]  # shape: (n_gal, n_samples, n_params)

# Check parameter order: [0]=logM*, [1]=logSFR, [2]=Av, [3]=[M/H] (if available)
logM_samples = posteriors[:, :, 0]
logSFR_samples = posteriors[:, :, 1]

# Compute sSFR = SFR/Mstar = 10**(logSFR-logM)
log_sSFR_samples = logSFR_samples - logM_samples
median_log_sSFR = np.nanmedian(log_sSFR_samples, axis=1)

plt.figure()
plt.scatter(median_log_sSFR, delta_logML, alpha=0.5)
plt.xlabel(r"median log(sSFR) (SBI)")
plt.ylabel(r"Δlog(M/L) (SBI - CWeb)")
plt.title(r"Δlog(M/L) vs median log(sSFR)")
plt.savefig(LOGDIR / "delta_logML_vs_median_log_sSFR.png")
plt.close()

# Av and [M/H] if available
if posteriors.shape[2] > 2:
    Av_samples = posteriors[:, :, 2]
    median_Av = np.nanmedian(Av_samples, axis=1)
    plt.figure()
    plt.scatter(median_Av, delta_logML, alpha=0.5)
    plt.xlabel(r"median Av (SBI)")
    plt.ylabel(r"Δlog(M/L) (SBI - CWeb)")
    plt.title(r"Δlog(M/L) vs median Av")
    plt.savefig(LOGDIR / "delta_logML_vs_median_Av.png")
    plt.close()

if posteriors.shape[2] > 3:
    Z_samples = posteriors[:, :, 3]
    median_Z = np.nanmedian(Z_samples, axis=1)
    plt.figure()
    plt.scatter(median_Z, delta_logML, alpha=0.5)
    plt.xlabel(r"median [M/H] (SBI)")
    plt.ylabel(r"Δlog(M/L) (SBI - CWeb)")
    plt.title(r"Δlog(M/L) vs median [M/H]")
    plt.savefig(LOGDIR / "delta_logML_vs_median_Z.png")
    plt.close()
