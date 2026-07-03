# EEG-Based Emotion Recognition & Classification (DEAP Dataset)

<div align="center">
  <a href="https://example.com">
    <img src="https://img.shields.io/badge/Run%20in%20Colab-orange?style=flat&logo=google-colab&logoColor=orange&label=%20Colab" alt="Run in Colab" />
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://www.python.org">
    <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python" alt="Python" />
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://www.gnu.org/licenses/agpl-3.0">
    <img src="https://img.shields.io/badge/License-AGPL%20v3-purple?logo=shield" alt="License: AGPL v3" />
  </a>
</div>

<br>

Final project for the **Human-Computer Interaction** course, Università degli Studi dell'Insubria. <br>
Authors: refer to [autori.csv](./autori.csv)


## Abstract

This project addresses binary recognition of **valence, arousal, dominance, and liking** (threshold at 5) from 32-channel EEG signals, using the **DEAP** dataset (Koelstra et al., 2012). Time-domain statistical features are extracted from four 15-second time segments (0–60 s) for each of the 40 trials across 32 subjects, and evaluated with five classifier families: SVM, KNN, Logistic Regression, Decision Tree, and LDA. The overall methodology parallels the segmented, PCA-based comparative approach of Doma & Pirouz (2020).

## Presentation

[Project presentation (Canva)](https://www.canva.com/design/DAHMZPbiJSo/PlByapyj98wxovXgj7Ux9w/view?utm_content=DAHMZPbiJSo&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h9024031fcf)

## Dataset

The dataset used is **DEAP** (*A Database for Emotion Analysis using Physiological Signals*), containing EEG and peripheral physiological recordings from 32 subjects while watching 40 music video clips, with self-report labels for valence, arousal, dominance, and liking. The dataset is not included in the repository (`deap-dataset.zip` is git-ignored) and must be downloaded separately and placed according to the structure expected by `load_data()`. It can be obtained from [Kaggle](https://www.kaggle.com/datasets/manh123df/deap-dataset).

Expected format for each subject file (`.dat`, pickle):
- `data`: array `40 × 40 × 8064` (trial × channel × sample)
- `labels`: array `40 × 4` (valence, arousal, dominance, liking)

## Pipeline

1. **Data loading** — `load_data()` reads the 32 `.dat` files (pickle, `latin1` encoding) and populates `subjects_data` / `subjects_labels`.
2. **Preprocessing** — `pre_process()` iterates over all **4 segments (0–15, 15–30, 30–45, 45–60 s)**. For each segment:
   - Parallel per-subject processing via `ProcessPoolExecutor` (worker count `round(os.cpu_count() * 0.80)`, one process per subject, 40 trials each)
   - Per trial: 4th-order Butterworth bandpass filter (0.5–45 Hz) → segment extraction (1920 samples, `segment_length = 15 * 128`) → 6 statistical features per channel (mean, standard deviation, variance, RMS, skewness, kurtosis) across 32 channels → 192-dimensional feature vector
   - Standardization (`StandardScaler`) and dimensionality reduction via **PCA with 20 components**
3. **Classification** — 5 classifiers evaluated on all **4 labels** (valence, arousal, dominance, liking) via `evaluate_metrics` using **10-fold cross-validation** (`cross_val_score`), taking the maximum value for each metric:

   | Classifier | Configuration |
   |---|---|
   | SVM | RBF kernel |
   | KNN | k=5 |
   | Logistic Regression | `max_iter=5000` |
   | Decision Tree | default parameters |
   | LDA | default parameters |

4. **Metrics** — accuracy, precision, recall, f1 (via `make_scorer`, `zero_division=0`), aggregated into a `DataFrame` and visualized with grouped bar charts per classifier (`plot_classifier_metrics`).

## Repository structure

```
.
├── Progetto_HCI.ipynb    # Main notebook (45 code cells)
├── export_python.py      # Pipeline exported as a script
├── requirements.txt      # Python dependencies
├── install.sh            # Automated install script
├── DEAP_dataset.pdf      # Koelstra et al. (2012) DEAP paper
└── LICENSE               # AGPL v3
```

## Installation and usage

**1. Automated install (recommended)**

   - For linux/MacOS
      ```bash
      curl -s https://raw.githubusercontent.com/lcavagnari/progetto_HCI/main/install.sh | sh
      ```

   - For windows 10/11
      ```powershell
      Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
      ```
      ```powershell
      iwr https://raw.githubusercontent.com/lcavagnari/progetto_HCI/main/install.ps1 | iex
      ```

**2. Manual install**

```bash
git clone https://github.com/lcavagnari/progetto_HCI.git

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```
**Dependencies** (`requirements.txt`): `numpy`, `scipy`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`, `jupyter`, `requests`.

**3. Usage**

Run the notebook
```bash
jupyter Progetto_HCI.ipynb
```
Run the standalone script
```bash
python3 export_python.py
```


## License

Distributed under the terms of the **GNU Affero General Public License v3.0** — see `LICENSE`.

## References

- Koelstra, S., Muhl, C., Soleymani, M., Lee, J.-S., Yazdani, A., Ebrahimi, T., Pun, T., Nijholt, A., & Patras, I. (2012). DEAP: A Database for Emotion Analysis Using Physiological Signals. *IEEE Transactions on Affective Computing*, 3(1), 18–31.ird-party mirror, not the official DEAP distribution.
- Doma, V., & Pirouz, M. (2020). A comparative analysis of machine learning methods for emotion recognition using EEG and peripheral physiological signals. *Journal of Big Data*, 7(1), 18.
- manh123df. *DEAP Dataset* [Data set]. Kaggle. https://www.kaggle.com/datasets/manh123df/deap-dataset — third-party mirror, not the official DEAP distribution.
