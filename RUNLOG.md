# Run log

Every atlas generation, training, and inference run gets **one row, added at the time of the run**.
Rules (learned the hard way from the v3 anomaly, 2026-06/07):

1. **Always pass `--train-seed`** to `train_euclid.py`. Unseeded runs are not publishable.
2. Record the **git commit** (`git rev-parse --short HEAD`) — commit before running, never run on dirty tree for keeper runs.
3. Record **md5 of the inputs** that are not in git: atlas, noise `.npy` set, model pickle
   (`md5sum library/<atlas> obs/obs_properties/mean_sigma_<prefix>.npy library/<model>`).
4. Record the **full command line** verbatim.
5. Record the headline metric (best val log-prob for training; Khostovan median Δ logM for inference).

| date | host | commit | stage | command (abridged) | inputs (md5 short) | seed | result |
|---|---|---|---|---|---|---|---|
| 2026-06-22 | server | ~d995272 | train | train_euclid, atlas_euclid_v3_100k, templfit, north_templfit(jwst Jun-4 gen) | model 067fd84b | none | val −0.8192 → model_euclid_v3.pkl |
| 2026-07-02 | local | d995272 | infer | inference khostovan total fluxes, model_euclid_v3 | — | — | **+0.209** (inference_khostovan_v3_total) |
| 2026-07-02 | server | — | train | v1 (north_total noise, atlas 50k ceil 12.5) | — | none | +0.314 (noise 2fwhm-inflated, superseded) |
| 2026-07-02 | server | — | train | v1_templnoise (north_templfit, atlas 50k ceil 12.5) | — | none | +0.280 |
| 2026-07-03 | local | — | train | v3atlas_local (exact v3 atlas + noise) | — | none | +0.369 |
| 2026-07-03 | local | — | train | v3atlas_JWnoise (v3 atlas + Jun-4 jwst noise, fingerprint-identical to v3 inputs) | — | none | +0.376 |
| 2026-07-05 | server | f94e2e4 | train | big_s1/s2 (15×500, seeds 1/2) | — | 1,2 | val −0.769/−0.743 → +0.369/+0.354 |
| 2026-07-06 | server | 990fbea | train | sw_big_lr1e4 (15×500, lr 1e-4, pat 60) | — | 1 | val −0.846 → +0.333 |
| 2026-07-06 | server | 990fbea | train | sw_lr1e4 (4×128, lr 1e-4) | — | 1 | val +1.44, COLLAPSED r≈0 |
| 2026-07-07 | local | 484e318 | forensic | compare_v3_vs_retrain + mock test (seed 777, 1000 mocks) | v3 067fd84b | 777 | retrain−v3 = +0.02 on mocks vs +0.17 on real data → off-manifold lottery, case closed |
| 2026-07-07 | server | 484e318 | train×5 | v1_s1..s5 (v3 atlas, 4×128, templfit noise) | — | 1–5 | s1/s3/s4 COLLAPSED (r≈0); s2 +0.366, s5 +0.392 |
| 2026-07-07 | local | 55b7c74* | calib | calibrate_zeropoints (fixed-z χ² fit of v3 atlas to Khostovan) | atlas v3 100k | — | per-band ZP ±0.03; VIS −0.07 @z0.5–1.5; **template-fit mass = +0.367 vs LePhare** → the +0.37 is the atlas mapping |
