"""
Shared plotting utilities for noise-model validation.
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _style():
	plt.rcParams.update({
		"axes.linewidth": 1.2,
		"font.size": 11,
		"axes.labelsize": 12,
		"legend.fontsize": 10,
		"figure.dpi": 120,
	})


def _compute_detection_fraction(x, detected, bins, min_count=25):
	total = np.histogram(x, bins=bins)[0].astype(float)
	det = np.histogram(x[detected], bins=bins)[0].astype(float)
	with np.errstate(divide="ignore", invalid="ignore"):
		frac = np.where(total >= min_count, det / total, np.nan)
	centers = 0.5 * (bins[:-1] + bins[1:])
	return centers, frac, total


def plot_sigma_vs_mag(fi, real_mag, real_sigma, mock_mag, mock_sigma,
					  mean_sigma_obs, percentiles, outdir,
					  filter_short, nondet_mag=99.0, mag_bright=16.0, mag_faint=30.0):
	_style()
	fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
	fig.suptitle(f"{filter_short[fi]}  —  σ vs magnitude", fontsize=13, fontweight="bold")

	det_r = np.isfinite(real_mag[fi]) & np.isfinite(real_sigma[fi])
	det_m = (mock_mag[fi] < nondet_mag - 0.5) & np.isfinite(mock_sigma[fi])

	for ax, lbl, m, s, color in [
		(axes[0], "Real data", real_mag[fi][det_r], real_sigma[fi][det_r], "#1f77b4"),
		(axes[1], "Mock (simulated)", mock_mag[fi][det_m], mock_sigma[fi][det_m], "#ff7f0e"),
	]:
		ax.scatter(m, s, s=1.5, alpha=0.3, color=color, rasterized=True)

		bins = percentiles[:, fi]
		n_bins = mean_sigma_obs.shape[1]
		edges = np.concatenate([[mag_bright], bins, [mag_faint]])
		for k in range(n_bins):
			ax.hlines(mean_sigma_obs[fi, k], edges[k], edges[k + 1],
					  colors="red", linewidths=2.0, label="Model mean" if k == 0 else "")
		ax.set_xlim(mag_bright, mag_faint)
		ax.set_ylim(0, 1.1)
		ax.set_xlabel("Magnitude (AB)")
		ax.set_ylabel("σ (mag)")
		ax.set_title(lbl)
		if k == 0:
			ax.legend(loc="upper left")

	if det_r.sum() > 10 and det_m.sum() > 10:
		from scipy.stats import ks_2samp

		s_r = real_sigma[fi][det_r]
		s_m = mock_sigma[fi][det_m]
		s_r = s_r[np.isfinite(s_r)]
		s_m = s_m[np.isfinite(s_m)]
		_, pval = ks_2samp(s_r, s_m)
		fig.text(0.5, 0.01, f"KS p-value (σ distribution): {pval:.3g}",
				 ha="center", fontsize=10, color="grey")

	plt.tight_layout(rect=[0, 0.04, 1, 1])
	out = Path(outdir) / f"sigma_vs_mag_{filter_short[fi].replace('/', '-')}.png"
	plt.savefig(out, bbox_inches="tight", dpi=150)
	plt.close()
	return out


def plot_mag_histogram(fi, real_mag, mock_mag, outdir,
					   filter_short, nondet_mag=99.0, mag_bright=16.0, mag_faint=30.0):
	_style()
	det_r = real_mag[fi][np.isfinite(real_mag[fi])]
	det_m = mock_mag[fi][mock_mag[fi] < nondet_mag - 0.5]

	if det_r.size == 0 and det_m.size == 0:
		return None

	fig, ax = plt.subplots(figsize=(7, 5))
	bins = np.linspace(mag_bright, mag_faint, 40)
	ax.hist(det_r, bins=bins, density=True, alpha=0.55, color="#1f77b4", label="Real")
	ax.hist(det_m, bins=bins, density=True, alpha=0.55, color="#ff7f0e", label="Mock")
	ax.set_xlabel("Magnitude (AB)")
	ax.set_ylabel("Density")
	ax.set_title(f"{filter_short[fi]}  —  Magnitude distribution (detected)")
	ax.legend()
	plt.tight_layout()
	out = Path(outdir) / f"mag_hist_{filter_short[fi].replace('/', '-')}.png"
	plt.savefig(out, bbox_inches="tight", dpi=150)
	plt.close()
	return out


def plot_true_mag_histogram(fi, real_mag, mock_true_mag, outdir,
							filter_short, mag_bright=16.0, mag_faint=30.0):
	_style()
	det_r = real_mag[fi][np.isfinite(real_mag[fi])]
	det_m = mock_true_mag[fi][np.isfinite(mock_true_mag[fi])]

	if det_r.size == 0 and det_m.size == 0:
		return None

	fig, ax = plt.subplots(figsize=(7, 5))
	bins = np.linspace(mag_bright, mag_faint, 100)
	ax.hist(det_r, bins=bins, density=True, alpha=0.55, color="#1f77b4", label="Real observed")
	ax.hist(det_m, bins=bins, density=True, alpha=0.55, color="#2ca02c", label="Mock true (pre-noise)")
	ax.set_xlabel("Magnitude (AB)")
	ax.set_ylabel("Density")
	ax.set_title(f"{filter_short[fi]}  —  True mock vs real magnitudes")
	ax.legend()
	plt.tight_layout()
	out = Path(outdir) / f"true_mag_hist_{filter_short[fi].replace('/', '-')}.png"
	plt.savefig(out, bbox_inches="tight", dpi=150)
	plt.close()
	return out


def plot_detection_fraction_vs_flux(fi, real_data, mock_data, limits, outdir,
									filter_short, snr_detection_threshold=2.0,
									nondet_mag=99.0):
	_style()
	limit_flux = limits[fi]
	real_flux = real_data["flux"][fi]
	real_err = real_data["err"][fi]
	real_valid = real_data["valid"][fi]
	real_snr = np.full_like(real_flux, np.nan, dtype=float)
	np.divide(real_flux, real_err, out=real_snr, where=real_valid)
	real_detected = real_valid & np.isfinite(real_snr) & (real_snr >= snr_detection_threshold) & (real_flux > 0)

	mock_flux = mock_data["true_flux"][fi]
	mock_mag = mock_data["mag"][fi]
	mock_sigma = mock_data["sigma"][fi]
	mock_snr = np.full_like(mock_sigma, np.nan, dtype=float)
	good_sigma = np.isfinite(mock_sigma) & (mock_sigma > 0)
	mock_snr[good_sigma] = 2.5 / (np.log(10) * mock_sigma[good_sigma])
	mock_detected = (mock_mag < nondet_mag - 0.5) & np.isfinite(mock_snr) & (mock_snr >= snr_detection_threshold)

	positive_real = real_flux[real_valid & (real_flux > 0)]
	positive_mock = mock_flux[np.isfinite(mock_flux) & (mock_flux > 0)]
	flux_min = np.nanpercentile(np.concatenate([positive_real, positive_mock]), 1)
	flux_max = np.nanpercentile(np.concatenate([positive_real, positive_mock]), 99.5)
	bins = np.geomspace(max(flux_min, 1e-4), max(flux_max, flux_min * 10), 30)

	centers_r, frac_r, _ = _compute_detection_fraction(real_flux[real_valid], real_detected[real_valid], bins)
	centers_m, frac_m, _ = _compute_detection_fraction(mock_flux[np.isfinite(mock_flux)], mock_detected[np.isfinite(mock_flux)], bins)

	fig, axes = plt.subplots(1, 2, figsize=(13, 5))
	fig.suptitle(f"{filter_short[fi]}  —  Detection diagnostics", fontsize=13, fontweight="bold")

	axes[0].step(centers_r, frac_r, where="mid", color="#1f77b4", linewidth=2, label="Real")
	axes[0].step(centers_m, frac_m, where="mid", color="#ff7f0e", linewidth=2, label="Mock")
	axes[0].axvline(limit_flux, color="gray", linestyle="--", linewidth=1, label="Adopted limit")
	axes[0].axhline(0.5, color="gray", linestyle="--", linewidth=1)
	axes[0].set_xscale("log")
	axes[0].set_xlabel("Flux (μJy)")
	axes[0].set_ylabel("Detection fraction")
	axes[0].set_title(f"Detection fraction vs flux (SNR ≥ {snr_detection_threshold:g})")
	axes[0].set_ylim(-0.05, 1.15)
	axes[0].legend(loc="lower right")

	bins_hist = np.geomspace(max(flux_min, 1e-4), max(flux_max, flux_min * 10), 40)
	axes[1].hist(real_flux[real_valid], bins=bins_hist, density=True, histtype="step", linewidth=2,
				 color="#1f77b4", label=f"Real all (n={real_valid.sum()})")
	axes[1].hist(mock_flux[np.isfinite(mock_flux)], bins=bins_hist, density=True, histtype="step", linewidth=2,
				 color="#ff7f0e", label=f"Mock all (n={np.isfinite(mock_flux).sum()})")
	axes[1].axvline(limit_flux, color="gray", linestyle="--", linewidth=1, label="Adopted limit")
	axes[1].set_xscale("log")
	axes[1].set_xlabel("Flux (μJy)")
	axes[1].set_ylabel("Density")
	axes[1].set_title("All-source flux distribution")
	axes[1].legend()

	plt.tight_layout()
	out = Path(outdir) / f"det_fraction_{filter_short[fi].replace('/', '-')}.png"
	plt.savefig(out, bbox_inches="tight", dpi=150)
	plt.close()
	return out


def plot_sigma_distribution(fi, real_sigma, mock_sigma, outdir,
							filter_short):
	_style()
	det_r = real_sigma[fi][np.isfinite(real_sigma[fi])]
	det_m = mock_sigma[fi][np.isfinite(mock_sigma[fi]) & (mock_sigma[fi] > 0)]
	det_m = det_m[det_m < 2.0]

	if det_r.size < 5 and det_m.size < 5:
		return None

	fig, ax = plt.subplots(figsize=(7, 5))
	bins = np.linspace(0, 1.2, 50)
	ax.hist(det_r, bins=bins, density=True, alpha=0.5, color="#1f77b4", label=f"Real  (n={det_r.size})")
	ax.hist(det_m, bins=bins, density=True, alpha=0.5, color="#ff7f0e", label=f"Mock  (n={det_m.size})")
	ax.set_xlabel("σ (mag)")
	ax.set_ylabel("Density")
	ax.set_title(f"{filter_short[fi]}  —  σ distribution (real vs mock)")
	ax.legend()

	if det_r.size > 10 and det_m.size > 10:
		from scipy.stats import ks_2samp

		stat, pval = ks_2samp(det_r, det_m)
		ax.text(0.97, 0.95, f"KS D={stat:.3f}\np={pval:.3g}",
				transform=ax.transAxes, ha="right", va="top",
				fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

	plt.tight_layout()
	out = Path(outdir) / f"sigma_dist_{filter_short[fi].replace('/', '-')}.png"
	plt.savefig(out, bbox_inches="tight", dpi=150)
	plt.close()
	return out


def plot_colors(real_mag, mock_mag, outdir,
				filter_short, color_pairs, nondet_mag=99.0):
	_style()
	n_colors = len(color_pairs)
	fig, axes = plt.subplots(n_colors, 2, figsize=(12, 4 * n_colors), sharex=True, sharey=True)
	axes = np.atleast_2d(axes)

	for idx, (ai, bi, label) in enumerate(color_pairs):
		det_r = np.isfinite(real_mag[ai]) & np.isfinite(real_mag[bi])
		color_r = real_mag[ai][det_r] - real_mag[bi][det_r]
		mag_r = real_mag[bi][det_r]

		det_m = (mock_mag[ai] < nondet_mag - 0.5) & (mock_mag[bi] < nondet_mag - 0.5)
		color_m = mock_mag[ai][det_m] - mock_mag[bi][det_m]
		mag_m = mock_mag[bi][det_m]

		ax_r = axes[idx, 0]
		ax_m = axes[idx, 1]

		if color_r.size > 0:
			ax_r.hexbin(mag_r, color_r, gridsize=50, bins="log", mincnt=1,
						cmap="Blues", linewidths=0)
		if color_m.size > 0:
			ax_m.hexbin(mag_m, color_m, gridsize=50, bins="log", mincnt=1,
						cmap="Oranges", linewidths=0)

		ax_r.set_xlim(18, 28)
		ax_r.set_ylim(-2, 4)
		ax_m.set_xlim(18, 28)
		ax_m.set_ylim(-2, 4)

		ax_r.set_ylabel(label)
		ax_r.set_title(f"{label} — Real")
		ax_m.set_title(f"{label} — Mock")

		if idx == n_colors - 1:
			ax_r.set_xlabel(f"{filter_short[bi]} (mag)")
			ax_m.set_xlabel(f"{filter_short[bi]} (mag)")

	fig.suptitle("Color diagnostics: Real (left) vs Mock (right)", fontsize=13, fontweight="bold")
	plt.tight_layout(rect=[0, 0, 1, 0.98])
	out = Path(outdir) / "colors.png"
	plt.savefig(out, bbox_inches="tight", dpi=150)
	plt.close()
	return out


def plot_sigma_vs_mag_grid(real_mag, real_sigma, mock_mag, mock_sigma,
						   mean_sigma_obs, percentiles, outdir,
						   filter_short, nondet_mag=99.0, mag_bright=16.0, mag_faint=30.0):
	_style()
	n_filt = len(filter_short)
	ncols = 5
	nrows = 2
	fig, axes = plt.subplots(nrows, ncols * 2, figsize=(28, 9),
							 gridspec_kw={"wspace": 0.05, "hspace": 0.4})

	for fi in range(n_filt):
		row = fi // ncols
		col_base = (fi % ncols) * 2

		for offset, (lbl, m, s, color) in enumerate([
			("Real", real_mag[fi], real_sigma[fi], "#1f77b4"),
			("Mock", mock_mag[fi], mock_sigma[fi], "#ff7f0e"),
		]):
			ax = axes[row, col_base + offset]
			if offset == 0:
				det = np.isfinite(m) & np.isfinite(s)
			else:
				det = (m < nondet_mag - 0.5) & np.isfinite(s)
			ax.scatter(m[det], s[det], s=1, alpha=0.2, color=color, rasterized=True)

			bins_f = percentiles[:, fi]
			n_bins = mean_sigma_obs.shape[1]
			edges = np.concatenate([[mag_bright], bins_f, [mag_faint]])
			for k in range(n_bins):
				ax.hlines(mean_sigma_obs[fi, k], edges[k], edges[k + 1],
						  colors="red", linewidths=1.5)

			ax.set_xlim(mag_bright, mag_faint)
			ax.set_ylim(0, 1.0)
			ax.set_title(f"{filter_short[fi]} — {lbl}", fontsize=8, pad=2)
			if col_base + offset == 0:
				ax.set_ylabel("σ (mag)", fontsize=8)
			if row == nrows - 1:
				ax.set_xlabel("Mag", fontsize=8)
			ax.tick_params(labelsize=7)

	fig.suptitle("σ vs Magnitude — all filters (Real | Mock)  ·  red = model mean",
				 fontsize=12, fontweight="bold")
	out = Path(outdir) / "sigma_vs_mag_ALL.png"
	plt.savefig(out, bbox_inches="tight", dpi=130)
	plt.close()
	return out
