[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18173783.svg)](https://doi.org/10.5281/zenodo.18173783)

# Quantifying Privacy Risks in Synthetic Data: A Study on Black-Box Membership Inference

This study evaluates the privacy risks associated with synthetic tabular data generation, specifically through black-box Membership Inference Attacks (MIAs). The experiments and methodologies used in the paper are implemented in this repository, ensuring replicability to the best extent possible.

## Replicability Measures

To facilitate reproducibility, we have included:
- The datasets used in the experiments.
- Instructions in `generators.py` to minimize randomness during synthetic data generation. However, due to the inherent stochastic nature of neural network training and potential hardware differences, achieving perfect determinism remains challenging.

## Datasets

The datasets used in this study are publicly available and can be downloaded from:
- **Adult Income Dataset**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult)
- **South German Credit**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/573/south+german+credit+update)
- **COMPAS Dataset**: [ProPublica](https://github.com/propublica/compas-analysis)

Each dataset underwent preprocessing to ensure consistency: duplicates and missing values were removed, and features typically excluded in standard machine learning pipelines, including identifiers that could link a record to an individual, were dropped.

## Reproducing Results

To reproduce the results from our study, follow these steps:

1. **Split the dataset into multiple parts:**
   ```bash
   python main.py -action=split -dataset="credit|adult|compas" -iterations=30
   ```
   This creates 30 splits of the dataset. Using the same seed ensures that the splits are always identical.

2. **Generate synthetic data:**
   ```bash
   python main.py -action=generate -generator="mixup|CTGAN|TVAE|CTAB-GAN+" -dataset="credit|adult|compas" -identifier=seq
   ```
   This generates synthetic data for each dataset split identified by `seq` using the specified generator and saves it.

3. **Perform Membership Inference Attacks (MIAs):**
   ```bash
   python main.py -action=attack -generator=mixup|CTGAN|TVAE|CTAB-GAN+ -dataset="credit|adult|compas" -identifier=seq
   ```
   This applies all MIAs to each dataset split identified by `seq` for the specified generator and saves the results.

4. **Evaluate privacy and utility:**
   ```bash
   python main.py -action=utility -generator=mixup|CTGAN|TVAE|CTAB-GAN+ -dataset="credit|adult|compas" -identifier=seq
   ```
   This evaluates both privacy and utility for each dataset split identified by `seq` and saves the results.

In our analysis, `seq` was always a list of numbers from 1 to 30 (e.g., `1,2,3,...,30`).

## Using CTAB-GAN+

We include **CTAB-GAN+** in this repository (under `CTABGAN`) as one of the synthetic data generators used in our study.

The original implementation is maintained at: https://github.com/Team-TUD/CTAB-GAN-Plus-DP
According to the project authors, CTAB-GAN+ DP is available under the **Apache License 2.0** (see [issue #3](https://github.com/Team-TUD/CTAB-GAN-Plus-DP/issues/3)).

### Modifications
To support reproducibility and forward compatibility, we applied the following changes:

- Updated the `__init__` function to allow configuring parameters such as **epochs**, **batch size**, and **discriminator steps**.
- Updated the usage of **BayesianGaussianMixture** in `transformer.py` to improve compatibility with newer versions of `scikit-learn`.

---

We appreciate your interest in our work and hope this repository helps advance research on synthetic data privacy!
