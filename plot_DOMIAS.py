from attack import DOMIAS
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ref_sizes = [0.1, 0.2, 0.3]
datasets = ['adult', 'credit', 'compas']
generators = ['mixup', 'TVAE', 'CTGAN', 'CTAB-GAN']

## code here and not in results_analysis.ipynb for speeding the process

colorblind_palette = sns.color_palette("colorblind", len(generators))

results = {dataset_name: {generator_name: {f'{ref}':[] for ref in ref_sizes} for generator_name in generators} for dataset_name in datasets}

for dataset_name in datasets:
    for generator_name in generators:
        
        for id in list(range(1, 31)):
            directory_name = f"{dataset_name}_{id}"
            directory_path = os.path.join("./split", directory_name)

            if not os.path.exists(directory_path):
                raise FileNotFoundError(f"For dataset {dataset_name} folder with identifier {id} does not exist inside split")

            train_file_path = os.path.join(directory_path, "train.csv")
            test_file_path = os.path.join(directory_path, "test.csv")
            synthetic_file_path = os.path.join('synthetic', generator_name, directory_name, "synthetic.csv")

            training_data = pd.read_csv(train_file_path)
            test_data = pd.read_csv(test_file_path)
            synthetic_data = pd.read_csv(synthetic_file_path)

            if dataset_name == 'adult':
                sample_size=4305
            elif dataset_name == 'credit':
                sample_size=160
            elif dataset_name == 'compas':
                sample_size=367
            
            for ref in ref_sizes:
                ref_size = int(len(training_data) * ref)

                attack = DOMIAS(training_data=training_data, test_data=test_data, synthetic_data=synthetic_data, reference_size=ref_size, sample_size=sample_size)
                results[dataset_name][generator_name][f'{ref}'].append(attack.perform_inference()['AUC-ROC'])

fig, axs = plt.subplots(1, 3, figsize=(15, 3))

for ax, dataset_name in zip(axs, datasets):
    for i, (generator_name, color) in enumerate(zip(generators, colorblind_palette)):
        means = []
        stds = []

        for ref in ref_sizes:
            res = results[dataset_name][generator_name][f'{ref}']
            means.append(np.mean(res))
            stds.append(np.std(res))

        print(f'For {dataset_name} {generator_name}:')
        for i in range(len(ref_sizes)):
            print(f'    {ref_sizes[i]}: {means[i]} +- {stds[i]}')

        ax.errorbar(
            ref_sizes,
            means,
            label=generator_name,
            color=color,
            capsize=3,
            linestyle='-',
            marker='o'
        )

    ax.set_title(dataset_name)
    ax.set_xlabel('Reference Size')
    ax.set_ylabel('AUC ROC')
    ax.set_xticks(ref_sizes)

    if dataset_name == 'compas':
        ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
    else:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

plt.savefig('plot_DOMIAS.pdf', format='pdf', bbox_inches='tight')