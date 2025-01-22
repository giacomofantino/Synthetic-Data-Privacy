# Quantifying Privacy Risks in Synthetic Data: A Study on Black-Box Membership Inference

This study evaluates the privacy risks associated with synthetic tabular data generation, specifically through black-box Membership Inference Attacks (MIAs). The experiments and methodologies used in the paper are implemented in this repository, ensuring replicability to the best extent possible.

## Replicability Measures

To facilitate reproducibility, we have included:
- The datasets used in the experiments.
- Instructions in `generators.py` to minimize randomness during synthetic data generation. However, due to the inherent stochastic nature of neural network training and potential hardware differences, achieving perfect determinism remains challenging.

## Datasets

The datasets used in this study are publicly available and can be downloaded from:
- **Adult Income Dataset**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult)
- **Give Me Some Credit Dataset**: [Kaggle](https://www.kaggle.com/competitions/GiveMeSomeCredit/data)
- **COMPAS Dataset**: [ProPublica](https://github.com/propublica/compas-analysis)

These datasets are provided in their original form and used in our experiments without modifications.

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

We employed **CTAB-GAN+** as one of the synthetic data generators in our study. However, due to the lack of a published license in its repository, we have decided not to include its source code in this repository to avoid legal issues.

To reproduce the results for CTAB-GAN+, you need to:
1. Download the **CTAB-GAN+** code from its original repository.
2. Place it in the designated folder within this repository.
3. Apply the following modifications:
   - Modify the `_init_` function to allow changes to parameters such as **epochs, batch size, and discriminator steps**.
   - Update **BayesianGaussianMixture** in `transformer.py` to ensure compatibility with future versions.

Once a license is added to the official CTAB-GAN+ repository, we plan to publish our modified version for better reproducibility.

---

We appreciate your interest in our work and hope this repository helps advance research on synthetic data privacy!

