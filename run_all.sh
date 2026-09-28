#!/usr/bin/env bash
# Reproduces every number and figure of the paper (about 25 min on one core).
set -e
mkdir -p results/figures
python3 src/canonical_results.py --out results      # b and its CI, Sigma, bed ratio, coronary exponent, species ratios
python3 src/metabolic_curvature.py --out results    # b under Kleiber, Rubner and curved metabolic laws
python3 src/amplitude_weight.py --out results       # weight of the pulsatile component
python3 src/feaver_rabbit.py --out results          # in vitro sensing laws; rabbit and mouse checks
python3 src/carotid_harmonics.py                    # young vs older carotid waveforms
python3 src/hr_prediction.py --out results          # heart-rate prediction
python3 src/make_pnas_figures.py                    # Figures 1-4
