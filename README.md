# InFlow: An Invariant Endothelial Flow Signal Across Species

Code and data to reproduce every number, figure and supplementary table of the manuscript *An invariant endothelial flow signal across species* (submitted to Journal of the Royal Society Interface). Version 1.

## Requirements

Python 3.10 or later. Tested with Python 3.12.3, numpy 2.4.4, scipy 1.17.1, pandas 3.0.2, matplotlib 3.10.8 and pillow 12.1.1.

```
pip install -r requirements.txt
```

## Run everything

From the repository root:

```
./run_all.sh
```

About 25 minutes on one core. Results are written to `results/` and figures to `results/figures/`.

## Run one analysis

| Command | Output | Used in the paper for |
|---|---|---|
| `python3 src/canonical_results.py --out results` | `canonical_results.tsv` | b and its 95% CI, variance removed by Σ, carotid/femoral ratio, coronary exponent, mouse and rabbit ratios |
| `python3 src/metabolic_curvature.py --out results` | `metabolic_curvature_b.tsv` | b under Kleiber, Rubner and curved metabolic laws; small vs large mammals |
| `python3 src/amplitude_weight.py --out results` | `amplitude_weight_summary.tsv`, `amplitude_weight_scan.tsv` | weight w of the pulsatile component |
| `python3 src/feaver_rabbit.py --out results` | `feaver_sensing_laws.tsv`, `rabbit_mouse_check.tsv` | in vitro sensing laws (Fig. 3); rabbit and mouse aorta |
| `python3 src/carotid_harmonics.py --out results` | `carotid_harmonics.tsv` | young vs older carotid waveforms (Fig. 4A–C) |
| `python3 src/hr_prediction.py --out results` | `hr_prediction.tsv` | heart-rate prediction (Fig. 4D) |
| `python3 src/make_figures.py` | `figures/rsif_fig1–4.png/.pdf/.tif` | Figures 1–4 (run the scripts above first) |
| `python3 src/make_esm.py --out results` | `figures/esm_figS1–S2`, `esm_tables.md` | supplementary figures S1–S2 and tables S2–S7 |

`systemic_allometry.py` and `sigma_exact_lib.py` are shared modules (arterial trees, exact Womersley factor).

## Data tables (`data/`)

| File | Content | Source |
|---|---|---|
| `seymour2019_points.tsv` | 92 paired flow and radius measurements, 20 arteries, 9 species, with source study | Seymour, Hu & Snelling, J Exp Biol 222:jeb199554 (2019), Table S1 |
| `feaver2013_tableS1.tsv` | harmonic amplitudes and NF-κB activity of 20 shear waveforms | Feaver, Gelfand & Blackman, Nat Commun 4:1525 (2013), Supplementary Table S1 |
| `carotid_waveforms_feature_points.tsv` | feature-point timings and amplitudes of carotid and vertebral flow waveforms | Ford et al., Physiol Meas 26:477 (2005), Table 2; Hoi et al., Physiol Meas 31:291 (2010), Table 2 |
| `reneman2009_table3_human_beds.tsv` | mean and peak wall shear stress in four human arteries | Reneman, Vink & Hoeks, Artery Res 3:73 (2009), Table 3 |
| `other_species_aorta.tsv` | mouse and rabbit aortic wall shear stress | Winter et al., J Cardiovasc Magn Reson (2019); Riemer et al., Ann Biomed Eng 48:1728 (2020) |

Values were transcribed from the published tables; cite the original sources when using them.

## Expected results

Key values produced by `./run_all.sh` (bootstrap seeds are fixed, so results are identical on every run):

| Quantity | File | Value |
|---|---|---|
| Maintenance exponent b (w = 0.4) | `canonical_results.tsv` | 0.675, 95% CI 0.550–0.725 |
| Variance of log wall shear stress removed by Σ | `canonical_results.tsv` | 59% |
| Carotid-to-femoral mean shear ratio | `canonical_results.tsv` | 2.8 (w = 0.4), 3.7 (w = 1) |
| Coronary flow–diameter exponent | `canonical_results.tsv` | 2.51 |
| b under Rubner and curved metabolic laws | `metabolic_curvature_b.tsv` | 0.675 |
| Pulsatile weight w | `amplitude_weight_summary.tsv` | 0.32–0.43 (in vitro, converted); 1.05 / 0.40 (young / older adults) |
| NF-κB fit, mean + w₁A₁ | `feaver_sensing_laws.tsv` | R² 0.878 (w₁ = 2.9) |
| Internal carotid first harmonic, young / older | `carotid_harmonics.tsv` | 0.25 / 0.43 of mean flow |
| Radius change for +20 bpm, aorta | `hr_prediction.tsv` | +4.0% |

## License

The code is released under the MIT License (`LICENSE`). The data tables are transcribed from published articles and are not covered by that license.
