# An Invariant Endothelial Flow Signal Across Species

Source code and data tables to reproduce every number and figure of the manuscript.

## Requirements

Python 3.10 or later.

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
| `python3 src/carotid_harmonics.py` | `carotid_harmonics.tsv` | young vs older carotid waveforms (Fig. 4A–C) |
| `python3 src/hr_prediction.py --out results` | `hr_prediction.tsv` | heart-rate prediction (Fig. 4D) |
| `python3 src/make_pnas_figures.py` | `figures/pnas_fig1–4.png/.pdf` | Figures 1–4 (run the scripts above first) |

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
