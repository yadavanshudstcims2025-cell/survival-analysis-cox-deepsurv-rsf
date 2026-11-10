# Neural Networks vs Cox Regression for Survival Analysis

A statistics project comparing the **Cox proportional hazards (PH) model**, **DeepSurv** (a neural network trained on the Cox partial likelihood) and a **random survival forest** for predicting time-to-event outcomes. The comparison is run in two ways:

1. a **simulation study** where the true model is known, so we can see *when* a neural network helps;
2. **two real breast cancer datasets** (GBSG and METABRIC) to see how much it helps in practice.

**Author:** Anshu Yadav, M.Sc. Statistics and Computing, DST-CIMS, Banaras Hindu University

The full write-up is in [`report/`](report/).

---

## Main findings

**Simulation** (4 scenarios, 200 replications each, 30% censoring, n = 1000):

| Scenario | Cox PH | DeepSurv | Oracle (true model) |
|---|---|---|---|
| A: PH, linear effects | 0.697 | 0.684 | 0.699 |
| B: PH, non-linear effects | 0.547 | 0.710 | 0.737 |
| C: non-PH, linear effects | 0.769 | 0.754 | 0.770 |
| D: non-PH, non-linear effects | 0.563 | 0.775 | 0.807 |

*(Harrell's C-index, higher is better.)*

- When the Cox model is correctly specified, it is essentially optimal and the networks are slightly worse.
- When the log-risk is non-linear, the linear Cox model collapses to near-chance ranking, while DeepSurv recovers most of the signal.
- Violating proportional hazards cost little for ranking in these scenarios.
- With only n = 300 training observations, the discrete-time neural model was unreliable.

**Real data** (5-fold cross-validation repeated 3 times, 15 splits):

| Dataset | Metric | Cox PH | DeepSurv | Random survival forest |
|---|---|---|---|---|
| GBSG (n = 2,232) | C-index | 0.662 | 0.670 | 0.674 |
| | Integrated Brier score (lower is better) | 0.1749 | 0.1705 | 0.1679 |
| METABRIC (n = 1,904) | C-index | 0.636 | 0.626 | 0.645 |
| | Integrated Brier score (lower is better) | 0.1831 | 0.1857 | 0.1785 |

- Gains over Cox in C-index were small and **not statistically significant** in either dataset.
- The random survival forest had the best mean on every metric in both datasets, and its improvement in integrated Brier score was significant after Holm adjustment (p = 0.012 in GBSG, p = 0.013 in METABRIC).
- DeepSurv improved the integrated Brier score in GBSG (Holm-adjusted p = 0.037) but not in METABRIC, where one of its 15 training runs also failed.
- Differences between models were tested with the corrected resampled t-test (Nadeau and Bengio, 2003), which accounts for overlapping training sets in cross-validation.

**Practical takeaway:** fit a Cox model first and diagnose it. Move to a flexible model when there is evidence of non-linear structure and enough data to support it.

---

## Repository structure

```
.
├── README.md
├── requirements.txt
├── report/
│   └── Survival_NN_Project_Report.pdf
├── simulation/
│   ├── survival_sim.py          # simulation study (NumPy only)
│   └── analysis.py              # tables, paired tests and figures from the simulation output
└── real_data/
    ├── Survival_RealData_Colab.ipynb   # real-data analysis (GBSG / METABRIC)
    └── results/
        ├── gbsg/                # CSV tables and PNG figures from the notebook
        └── metabric/
```

Datasets are **not** stored in this repository. The notebook downloads them automatically through the `pycox` package.

---

## How to run

### 1. Simulation study

Needs only Python 3, NumPy, SciPy, pandas and Matplotlib.

```bash
pip install numpy scipy pandas matplotlib

# main experiment: n = 1000, 200 replications per scenario (about 3-4 minutes on 2 cores)
python simulation/survival_sim.py 200

# sample-size experiment: n = 300 and n = 3000, 60 replications each
SIM_N=300  python simulation/survival_sim.py 60
SIM_N=3000 python simulation/survival_sim.py 60

# build the tables, paired tests and figures
python simulation/analysis.py
```

All random seeds are fixed, so the results are reproducible. The models (Cox PH, DeepSurv, and a discrete-time neural model) are implemented from scratch in NumPy, including back-propagation. Gradients were checked against finite differences.

### 2. Real data (Google Colab recommended)

1. Open `real_data/Survival_RealData_Colab.ipynb` in [Google Colab](https://colab.research.google.com).
2. Run the install cell. If Colab asks to restart the session, restart it and run from the second cell onwards.
3. Choose the dataset in the settings cell: `DATASET = "gbsg"` or `DATASET = "metabric"`.
4. Run all cells. Tables and figures are saved and downloaded at the end (about 5 to 10 minutes on a CPU runtime).

The notebook uses `pycox`, `lifelines`, `scikit-survival` and `torchtuples`, and produces:
Kaplan-Meier curves, Cox hazard ratios, a proportional hazards test, cross-validated C-index, time-dependent AUC and integrated Brier score, paired model comparisons, a calibration plot, and permutation importance.

Note: the notebook overwrites its output files each time, so move the results of one dataset into its own folder before running the next.

---

## Methods in brief

- **Models:** Cox PH (lifelines), DeepSurv (pycox / PyTorch, two hidden layers of 32 units), random survival forest (scikit-survival, 300 trees).
- **Metrics:** Harrell's C-index, time-dependent AUC, integrated Brier score (inverse-probability-of-censoring weighted).
- **Simulation:** 6 correlated covariates (2 pure noise); Weibull PH or log-normal AFT event times; linear or non-linear log-risk; independent exponential censoring calibrated to 30%. Paired comparisons use bootstrap intervals and Wilcoxon tests.
- **Real data:** repeated stratified cross-validation, corrected resampled t-test, Holm adjustment over the C-index and Brier-score comparisons, and a Bonferroni-corrected Schoenfeld-residual test for proportional hazards.
- Hyperparameters were fixed in advance and **not tuned**.

---

## Limitations

- Only two real datasets, so the real-data findings illustrate rather than generalise.
- Hyperparameters were not tuned; tuning might change the gaps between models.
- The Cox baseline is the plain linear model, without splines or interactions, so the comparison shows what a network gives automatically, not what it adds beyond a well-built Cox model.
- In the METABRIC run, AUC and Brier score could not be computed in 3 of 15 splits (the same splits for all models; cause not diagnosed), so those comparisons use 12 splits. One DeepSurv fit on METABRIC also failed; the report shows results with and without it.
- The corrected resampled t-test is an approximation based on overlapping splits, so p-values close to 0.05 should not be over-interpreted.
- Simulation scenarios do not include crossing survival curves or time-varying effects.

---

## References

- Katzman et al. (2018). DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network. *BMC Medical Research Methodology*, 18, 24.
- Kvamme, Borgan and Scheel (2019). Time-to-event prediction with neural networks and Cox regression. *JMLR*, 20(129).
- Ishwaran et al. (2008). Random survival forests. *Annals of Applied Statistics*, 2(3).
- Nadeau and Bengio (2003). Inference for the generalization error. *Machine Learning*, 52.
- Davidson-Pilon (2019). lifelines. *Journal of Open Source Software*, 4(40).
- Pölsterl (2020). scikit-survival. *JMLR*, 21(212).

The full reference list is in the report.

---

## License

Add a `LICENSE` file (for example MIT) if you want others to reuse the code.
