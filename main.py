import argparse
import os
import pandas as pd
from generators import MixupDataGenerator, CTGANDataGenerator, PATEGANDataGenerator, SDVDataGenerator
import shutil
from process import DataPreprocessor
from attack import DistanceBasedMembershipInference, DistributionBasedMembershipInference, MonteCarloMembershipInference, DOMIAS, PrivacyMetricsEvaluator
from utility import UtilityEvaluator
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

dataset_folder="./dataset"
split_dir="./split"

#each split is associated with a seed, so we can obtain the same splits again
#if more than 30 splits needed add new seeds
split_seeds = [5305744, 5680651, 7972578, 18034864, 97736577, 154830022, 183746029, 231614367, 255564838, 291458166, 302651695, 311380335, 353704319, 392090150, 472537969, 476408914, 497060577, 530757929, 543799220, 551330140, 556836201, 685879090, 686965655, 697331522, 786750199, 794988448, 877221475, 906137553, 930979619, 956886724]

def split_dataset(dataset_name, label, iterations, split_ratio=0.8):
    """
    Function to split the dataset into training and testing sets.
    
    Parameters:
    - dataset_name: str, name of the dataset to split
    - split_ratio: float, ratio of the dataset to use for training (default is 0.8)
    
    Returns:
    - train_set: training dataset
    - test_set: testing dataset
    """
    dataset_path = os.path.join(dataset_folder, f"{dataset_name}.csv")

    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset '{dataset_name}.csv' not found in folder '{dataset_folder}'.")
    
    df = pd.read_csv(dataset_path)

    if dataset_name == "adult":
        df = df.drop('fnlwgt',axis=1)
        df = df.drop('education',axis=1) #education and education_num are in a one to one relationship
        df = df[~df.isin([' ?']).any(axis=1)]
        df = df.drop_duplicates().reset_index(drop=True)


    #apply the encoding of the dataset
    dp = DataPreprocessor()
    df_enc = dp.fit_transform(df)

    for i in range(1, iterations+1):
        # Shuffle the dataset
        df_enc = df_enc.sample(frac=1, random_state=split_seeds[i-1]).reset_index(drop=True)

        # Split the dataset, keeping balanced classes
        train_set, test_set = train_test_split(df_enc, train_size=split_ratio, stratify=df_enc[label], random_state=split_seeds[i-1])

        #reference set for DOMIAS
        reference_set = train_set[:2000]

        base_folder_name = f"{dataset_name}"

        new_folder_name = f"{base_folder_name}_{i}"
        new_folder_path = os.path.join(split_dir, new_folder_name)

        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)

        # Save the train and test sets as CSV files
        train_file_path = os.path.join(new_folder_path, f"train.csv")
        test_file_path = os.path.join(new_folder_path, f"test.csv")
        reference_file_path = os.path.join(new_folder_path, f"reference.csv")

        train_set.to_csv(train_file_path, index=False)
        test_set.to_csv(test_file_path, index=False)
        reference_set.to_csv(reference_file_path, index=False)

        #save encoding
        dp.save_label_encoders(new_folder_path)

        print(f"Dataset '{dataset_name}' has been split and saved in '{new_folder_path}'.")

    return

def generate_synthetic_data(generator_name, dataset_name, num_samples, identifier):
    """
    Function to generate synthetic data.
    
    Parameters:
    - generator_name: str, name of the synthetic data generator
    - dataset_name: str, name of the dataset to generate data for
    - num_samples: int, number of synthetic samples to generate
    
    Returns:
    - synthetic_data: generated synthetic dataset
    """

    #save information of the features based on the dataset
    if dataset_name == "adult":
        integer_columns = ['age','workclass','education_num','marital_status','occupation','relationship','race','sex','capital_loss','hours_per_week','native_country','income_class']
        
        #for CTGAN
        discrete_columns = [
            'workclass',
            'education_num',
            'marital_status',
            'occupation',
            'relationship',
            'race',
            'sex',
            'native_country',
            'income_class'
        ]
        
        epochs = 50
    elif dataset_name == "titanic":
        integer_columns = ['Survived','Pclass','Sex','Age','SibSp','Parch','Ticket','Cabin','Embarked']
        epochs=100

    # Instantiate the generator
    for id in identifier:
        if generator_name == 'mixup':
            generator = MixupDataGenerator(integer_columns)
        elif generator_name == 'mixup_privacy':
            generator = MixupDataGenerator(integer_columns, alpha=1)  # TODO: consider a better value eventually (possibly use beta)
        elif generator_name == 'CTGAN':
            generator = CTGANDataGenerator(epochs=epochs, discrete_columns=discrete_columns)
        elif generator_name == 'PATE-GAN':
            generator = PATEGANDataGenerator(epochs=epochs)
        elif generator_name == 'SDV':
            generator = SDVDataGenerator()
        else:
            raise ValueError(f"Generator '{generator_name}' is not recognized.")
    
        directory_name = f"{dataset_name}_{id}"
        directory_path = os.path.join(split_dir, directory_name)

        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"For dataset {dataset_name} folder with identifier {id} does not exist inside split")

        train_file_path = os.path.join(directory_path, "train.csv")

        if not os.path.isfile(train_file_path):
            raise FileNotFoundError(f"The file {train_file_path} does not exist.")

        # Load the training data
        training_data = pd.read_csv(train_file_path)
        
        # Train the generator with the training data
        epoch_loss_df = generator.train(training_data)
        
        #if num_samples is stop specified use the size of the original data
        if num_samples is None:
            num_samples = len(training_data)

        # Generate synthetic data
        synthetic_data = generator.generate(num_samples)  # Generate the same number of samples as training data
        
        # We can assume a post-processing operation to correct some data according to the semantic
        if dataset_name == 'adult':
            synthetic_data['capital_gain'] = synthetic_data['capital_gain'].apply(lambda x: max(x, 0))
            synthetic_data['capital_loss'] = synthetic_data['capital_loss'].apply(lambda x: max(x, 0))
            synthetic_data['age'] = synthetic_data['age'].clip(17, 90)
            synthetic_data['hours_per_week'] = synthetic_data['hours_per_week'].clip(1, 99)
            synthetic_data['education_num'] = synthetic_data['education_num'].clip(1, 16)
            
        output_dir = os.path.join('synthetic', generator_name, directory_name)

        # If the synthetic directory already exists, remove it
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        # Create output directory for synthetic data
        os.makedirs(output_dir)
        
        # Save the synthetic data to a CSV file
        output_file = os.path.join(output_dir, "synthetic.csv")
        synthetic_data.to_csv(output_file, index=False)
        
        print(f"Generated {len(synthetic_data)} samples for '{dataset_name}_{id}' and saved to '{output_file}'.")
    return


def perform_attack(generator_name, dataset_name, identifier):
    """
    Function to perform an attack on the dataset.
    
    Parameters:
    - attack_type: str, type of attack to perform
    - dataset_name: str, name of the dataset to attack
    
    Returns:
    - attack_result: result of the attack
    """

    for id in identifier:
        directory_name = f"{dataset_name}_{id}"
        directory_path = os.path.join(split_dir, directory_name)

        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"For dataset {dataset_name} folder with identifier {id} does not exist inside split")

        train_file_path = os.path.join(directory_path, "train.csv")
        test_file_path = os.path.join(directory_path, "test.csv")
        reference_file_path = os.path.join(directory_path, "reference.csv")
        synthetic_file_path = os.path.join('synthetic', generator_name, directory_name, "synthetic.csv")

        if not os.path.isfile(train_file_path):
            raise FileNotFoundError(f"The file {train_file_path} does not exist.")
        
        if not os.path.isfile(synthetic_file_path):
            raise FileNotFoundError(f"The file {synthetic_file_path} does not exist.")
        
        # Load the data
        training_data = pd.read_csv(train_file_path)
        test_data = pd.read_csv(test_file_path)
        reference_data = pd.read_csv(reference_file_path)
        synthetic_data = pd.read_csv(synthetic_file_path)

        attacks= {
            "Distance_MIA":DistanceBasedMembershipInference(training_data=training_data, test_data=test_data, synthetic_data=synthetic_data),
            "Distribution_MIA":DistributionBasedMembershipInference(training_data=training_data, test_data=test_data, synthetic_data=synthetic_data),
            "MonteCarlo_MIA":MonteCarloMembershipInference(training_data=training_data, test_data=test_data, synthetic_data=synthetic_data),
            "DOMIAS":DOMIAS(training_data=training_data, test_data=test_data, synthetic_data=synthetic_data, reference_data=reference_data)
        }

        for name, attacker in attacks.items():
            result = attacker.perform_inference()
            print(f"For {dataset_name}_{id} attack {name} AUC ROC was {result['AUC-ROC']}")
    return

def evaluate_utility(generator_name, dataset_name, identifier):
    if dataset_name == "adult":
        label = 'income_class'
    
    for id in identifier:
        directory_name = f"{dataset_name}_{id}"
        directory_path = os.path.join(split_dir, directory_name)

        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"For dataset {dataset_name} folder with identifier {id} does not exist inside split")

        train_file_path = os.path.join(directory_path, "train.csv")
        test_file_path = os.path.join(directory_path, "test.csv")
        synthetic_file_path = os.path.join('synthetic', generator_name, directory_name, "synthetic.csv")

        if not os.path.isfile(train_file_path):
            raise FileNotFoundError(f"The file {train_file_path} does not exist.")
        
        if not os.path.isfile(synthetic_file_path):
            raise FileNotFoundError(f"The file {synthetic_file_path} does not exist.")
        
        # Load the data
        training_data = pd.read_csv(train_file_path)
        test_data = pd.read_csv(test_file_path)
        synthetic_data = pd.read_csv(synthetic_file_path)

        evaluator = UtilityEvaluator(training_data=training_data, test_data=test_data, synthetic_data=synthetic_data)
        privacy = PrivacyMetricsEvaluator(training_data=training_data, test_data=test_data, synthetic_data=synthetic_data)

        results, results_train = evaluator.evaluate(label)
        privacy_results = privacy.compute_privacy_metrics()
        
        print(f"For {dataset_name}_{id} difference is")
        print(results_train)
        # Display the flattened DataFrame

        
        print(f"Statistical values: JSD {results['JSD']}, WSD {results['WSD']} diff corr {results['corr-dist']}")

        #TODO this is just a first implementation, will decide what to really do
        mean_dcr_synthetic = np.mean(privacy_results['DCR_Synth'])
        mean_dcr_test = np.mean(privacy_results['DCR_Test'])
        mean_nndr_synthetic = np.mean(privacy_results['NNDR_Synth'])
        mean_nndr_test = np.mean(privacy_results['NNDR_Test'])

        print()
        print(f'Privacy metrics for synthetic data are DCR: {mean_dcr_synthetic} and NNDR: {mean_nndr_synthetic}')
        print(f'Privacy metrics for test data are DCR: {mean_dcr_test} and NNDR: {mean_nndr_test}')
        print(f'Compared to test we have a difference of DCR: {mean_dcr_test - mean_dcr_synthetic} and NNDR: {mean_nndr_test - mean_nndr_synthetic}')
        print()
    return


def main(action, dataset_name, **kwargs):
    """
    Main function to perform the specified action.
    
    Parameters:
    - action: str, action to perform ('split', 'generate', 'attack', 'utility')
    - dataset_name: str, name of the dataset
    - kwargs: additional parameters for the action
    """

    if action != 'split':
        #obtain the list of identifiers
        identifier_s = kwargs.get('identifier')
        identifier = list(map(int, identifier_s.split(',')))

    if action == 'split':
        if dataset_name == 'adult':
            label='income_class'

        split_ratio = kwargs.get('split_ratio', 0.8)
        iterations = kwargs.get('iterations', 1)
        split_dataset(dataset_name, label, iterations, split_ratio=split_ratio)
        return 
    
    elif action == 'generate':
        generator_name = kwargs.get('generator_name')
        num_samples = kwargs.get('num_samples', None)
        return generate_synthetic_data(generator_name, dataset_name, num_samples, identifier)
    
    elif action == 'attack':
        generator_name = kwargs.get('generator_name')
        return perform_attack(generator_name, dataset_name, identifier)
    
    elif action == 'utility':
        generator_name = kwargs.get('generator_name')
        return evaluate_utility(generator_name, dataset_name, identifier)
    else:
        raise ValueError("Invalid action. Choose from 'split', 'generate', 'utility', or 'attack'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform actions on datasets.")
    parser.add_argument('-action', type=str, required=True, help="Action to perform: 'split', 'generate', 'utility', or 'attack'")
    parser.add_argument('-dataset', type=str, required=True, help="Name of the dataset")
    
    # Optional arguments for each action
    parser.add_argument('-split_ratio', type=float, default=0.8, help="Split ratio for dataset splitting (default: 0.8)")
    parser.add_argument('-generator', type=str, help="Name of the synthetic data generator (required for 'generate' action)")
    parser.add_argument('-num_samples', type=int, default=None, help="Number of samples to generate (default: 100)")
    parser.add_argument('-identifier', type=str, help="Identifier of the split or synthetic data for the generate or attack actions")
    parser.add_argument('-iterations', type=int, help="Number of iterations for the action split")

    args = parser.parse_args()

    # Validate and call the main function
    if args.action == 'generate' and not args.generator:
        parser.error("The '-generator' argument is required for the 'generate' action.")
    if args.action == 'attack' and not args.generator:
        parser.error("The '-generator' argument is required for the 'attack' action.")
    if (args.action == 'attack' or args.action == 'generate') and not args.identifier:
        parser.error(f"Need an identifier for performing the action {args.action}")

    main(args.action, args.dataset, 
         split_ratio=args.split_ratio, 
         generator_name=args.generator, 
         num_samples=args.num_samples,
         identifier=args.identifier,
         iterations=args.iterations)
