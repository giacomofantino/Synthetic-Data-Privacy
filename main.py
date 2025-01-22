import argparse
import os
import pandas as pd
from generators import MixupDataGenerator, CTGANDataGenerator, TVAEDataGenerator, CTABGANDataGenerator
import shutil
import gc
from process import DataPreprocessor
from attack import DistanceBasedMembershipInference, DistributionBasedMembershipInference, MonteCarloMembershipInference, DOMIAS ,PrivacyMetricsEvaluator
from utility import UtilityEvaluator
from sklearn.model_selection import train_test_split

dataset_folder="./dataset"
split_dir="./split"

#each split is associated with a seed, so we can obtain the same splits again
#if more than 30 splits needed add new seeds
split_seeds = [5305744, 5680651, 7972578, 18034864, 97736577, 154830022, 183746029, 231614367, 255564838, 291458166, 302651695, 311380335, 353704319, 392090150, 472537969, 476408914, 497060577, 530757929, 543799220, 551330140, 556836201, 685879090, 686965655, 697331522, 786750199, 794988448, 877221475, 906137553, 930979619, 956886724]

def split_dataset(dataset_name, label, iterations, split_ratio=0.8):
    """
    Function to split the dataset into training and testing sets.
    """
    dataset_path = os.path.join(dataset_folder, f"{dataset_name}.csv")

    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset '{dataset_name}.csv' not found in folder '{dataset_folder}'.")
    
    if dataset_name == "adult":
        separator = ','
    elif dataset_name == "credit":
        separator = ' '
    elif dataset_name == "compas":
        separator = ','
    else:
        ValueError("Dataset not recognized")

    df = pd.read_csv(dataset_path, sep=separator)

    if dataset_name == "adult":
        df = df.drop('fnlwgt',axis=1)
        df = df.drop('education',axis=1) #education and education_num are in a one to one relationship
        df = df[~df.isin([' ?']).any(axis=1)]
        df = df.drop_duplicates().reset_index(drop=True)
    elif dataset_name == "credit":
        df = df.drop_duplicates().reset_index(drop=True)
        df = df.dropna()
    elif dataset_name == "compas":
        df = df[['c_charge_degree',	'race',	'age_cat',	'score_text', 'sex', 'priors_count', 'days_b_screening_arrest',	'decile_score',	'two_year_recid']]
        
        df = df[df['days_b_screening_arrest'] <= 30]
        df = df[df['days_b_screening_arrest'] >= -30]
        # days_b_screening_arrest can be converted to integer
        df['days_b_screening_arrest'] = df['days_b_screening_arrest'].astype(int)
        df = df[df['c_charge_degree'] != 'O'] #ordinary traffic offenses removed since don't result in jail time
        df = df[df['score_text'] != 'N/A']

        df = df.dropna()
        df = df.drop_duplicates().reset_index(drop=True)

    #apply the encoding of the dataset
    dp = DataPreprocessor()
    df_enc = dp.fit_transform(df)

    for i in range(1, iterations+1):
        # Shuffle the dataset
        df_enc = df_enc.sample(frac=1, random_state=split_seeds[i-1]).reset_index(drop=True)

        # Split the dataset, keeping balanced classes
        train_set, test_set = train_test_split(df_enc, train_size=split_ratio, stratify=df_enc[label], random_state=split_seeds[i-1])

        base_folder_name = f"{dataset_name}"

        new_folder_name = f"{base_folder_name}_{i}"
        new_folder_path = os.path.join(split_dir, new_folder_name)

        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)

        # Save the train and test sets as CSV files
        train_file_path = os.path.join(new_folder_path, f"train.csv")
        test_file_path = os.path.join(new_folder_path, f"test.csv")

        train_set.to_csv(train_file_path, index=False)
        test_set.to_csv(test_file_path, index=False)

        #save encoding
        dp.save_label_encoders(new_folder_path)

        print(f"Dataset '{dataset_name}' has been split and saved in '{new_folder_path}'.")

    return

def generate_synthetic_data(generator_name, dataset_name, num_samples, identifier):
    """
    Function to generate synthetic data.
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
        discriminator_steps = 1
        batch_size = 500
        epochs_ctgan = 150
        epochs_TVAE = 300
        epochs_ctabgan = 50

        ##needed for CTAB-GAN
        mixed_columns= {'capital_loss':[0.0],'capital_gain':[0.0]}
        general_columns = ["age"]
        problem_type={"Classification": "income_class"}
        log_columns = []
    elif dataset_name == "credit":
        integer_columns = ['laufkont', 'laufzeit', 'moral', 'verw', 'hoehe', 'sparkont', 'beszeit', 'rate', 'famges', 'buerge', 'wohnzeit',  'verm', 'alter', 'weitkred', 'wohn', 'bishkred', 'beruf', 'pers', 'telef', 'gastarb', 'kredit']
        
        discrete_columns = [
            'laufkont',    # Status of the checking account
            'moral',       # Credit history
            'verw',        # Purpose of the credit
            'sparkont',    # Debtor's savings
            'beszeit',     # Duration of employment
            'famges',      # Combined marital status and sex
            'buerge',      # Other debtors or guarantors
            'verm',        # Debtor's most valuable property
            'wohn',        # Type of housing
            'bishkred',    # Number of credits at the bank
            'beruf',       # Quality of the debtor's job
            'kredit'       # Credit risk (target variable)
        ]

        discriminator_steps = 3
        batch_size = 100
        epochs_ctgan = 150
        epochs_TVAE = 250
        epochs_ctabgan = 150

        ##needed for CTAB-GAN
        mixed_columns= {}
        general_columns = ["alter"]
        problem_type={"Classification": "kredit"}
        log_columns = []
    
    elif dataset_name == "compas":
        integer_columns = ['c_charge_degree','race','age_cat','score_text','sex','priors_count','days_b_screening_arrest','decile_score','two_year_recid']

        discrete_columns = [
            'race',
            'age_cat',
            'score_text',
            'sex',
            'c_charge_degree',
            'decile_score',
            'two_year_recid']

        discriminator_steps = 1
        batch_size = 100
        epochs_ctgan = 200
        epochs_TVAE = 250
        epochs_ctabgan = 50

        ##needed for CTAB-GAN
        mixed_columns = {"days_b_screening_arrest":[-1, 0]}
        general_columns = []
        problem_type={"Classification": "two_year_recid"}
        log_columns = ["priors_count"]

    # Instantiate the generator
    for id in identifier:
        #load the training data
        directory_name = f"{dataset_name}_{id}"
        directory_path = os.path.join(split_dir, directory_name)

        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"For dataset {dataset_name} folder with identifier {id} does not exist inside split")

        train_file_path = os.path.join(directory_path, "train.csv")

        if not os.path.isfile(train_file_path):
            raise FileNotFoundError(f"The file {train_file_path} does not exist.")
            
        if generator_name == 'mixup':
            generator = MixupDataGenerator(integer_columns)
        elif generator_name == 'CTGAN':
            generator = CTGANDataGenerator(epochs=epochs_ctgan, discrete_columns=discrete_columns,batch_size=batch_size, discriminator_steps=discriminator_steps)
        elif generator_name == 'TVAE':
            generator = TVAEDataGenerator(epochs=epochs_TVAE, batch_size=batch_size)
        elif generator_name == 'CTAB-GAN':
            generator = CTABGANDataGenerator(train_file_path,epochs=epochs_ctabgan, integer_columns=integer_columns, categorical_columns=discrete_columns,
                                             batch_size=batch_size, mixed_columns=mixed_columns, general_columns=general_columns, problem_type=problem_type,
                                             log_columns=log_columns)
        else:
            raise ValueError(f"Generator '{generator_name}' is not recognized.")

        # Load the training data
        training_data = pd.read_csv(train_file_path)
        
        # Train the generator with the training data
        generator.train(training_data)
        
        #if num_samples is stop specified use the size of the original data
        if num_samples is None:
            num_samples = len(training_data)

        # Generate synthetic data
        synthetic_data = generator.generate(num_samples)  # Generate the same number of samples as training data
        
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
    """
    results = []
    output_dir = "results/"

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

        #values for 20% and 10% of training data
        if dataset_name == 'adult':
            sample_size=4305
            ref_size=2152
        elif dataset_name == 'credit':
            sample_size=160
            ref_size=80
        elif dataset_name == 'compas':
            sample_size=367
            ref_size=184
        else:
            ValueError("Dataset not recognized")

        attacks= {
            "Distance_MIA":DistanceBasedMembershipInference(training_data=training_data, test_data=test_data, synthetic_data=synthetic_data, sample_size=sample_size),
            "Distribution_MIA":DistributionBasedMembershipInference(training_data=training_data, test_data=test_data, synthetic_data=synthetic_data, sample_size=sample_size),
            "MonteCarlo_MIA":MonteCarloMembershipInference(training_data=training_data, test_data=test_data, synthetic_data=synthetic_data, sample_size=sample_size),
            "DOMIAS":DOMIAS(training_data=training_data, test_data=test_data, synthetic_data=synthetic_data, reference_size=ref_size, sample_size=sample_size),
        }
        
        # Perform attacks and write results
        for name, attacker in attacks.items():
            result = attacker.perform_inference()
            auc_roc = result['AUC-ROC']
            
            #print(f"Attack '{name}' on '{dataset_name}_{id}' has AUC-ROC: {auc_roc}")
            results.append([f"{dataset_name}_{id}",generator_name, name, auc_roc])

    df = pd.DataFrame(results, columns=["Dataset_ID", "Generator", "Attack_Name", "AUC_ROC"])
    output_file = os.path.join(output_dir, f"output_attack_{dataset_name}_{generator_name}.csv")
    df.to_csv(output_file, index=False)
    print(f'Attack results saved to {output_file}')
    return

def evaluate_utility(generator_name, dataset_name, identifier):
    if dataset_name == "adult":
        label = 'income_class'
    elif dataset_name == "credit":
        label = 'kredit'
    elif dataset_name == "compas":
        label = 'two_year_recid'
    else:
        ValueError("Dataset not recognized")
    
    utility_list = []
    privacy_list = []

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

        evaluator = UtilityEvaluator(training_data=training_data, test_data=test_data, synthetic_data=synthetic_data, label=label)
        privacy = PrivacyMetricsEvaluator(training_data=training_data, test_data=test_data, synthetic_data=synthetic_data)

        diff_accuracy, diff_f1, diff_auc = evaluator.evaluate()

        utility_list.append([f"{dataset_name}_{id}", generator_name, diff_accuracy, diff_f1, diff_auc])

        privacy_results = privacy.compute_privacy_metrics()

        privacy_list.append([f"{dataset_name}_{id}", generator_name, privacy_results['DCR_Synth'], privacy_results['NNDR_Synth'], privacy_results['Loss']])
        
        ## since computation of privacy is heavy, may need to explicity call the garbage collector
        del training_data, test_data, synthetic_data, privacy, privacy_results
        gc.collect()

        print(f'Finished {dataset_name}_{id}')
    df_utility = pd.DataFrame(utility_list, columns=["Dataset_ID", "Generator", "Diff_Accuracy", "Diff_F1", "Diff_AUC"])
    df_privacy = pd.DataFrame(privacy_list, columns=["Dataset_ID", "Generator", "DCR","NNDR", "Privacy Loss"])

    df_utility.to_csv(f"results/output_utility_{dataset_name}_{generator_name}.csv", index=False)
    df_privacy.to_csv(f"results/output_privacy_{dataset_name}_{generator_name}.csv", index=False)
    return


def main(action, dataset_name, **kwargs):
    """
    Main function to perform the specified action.
    """

    if action != 'split':
        #obtain the list of identifiers
        identifier_s = kwargs.get('identifier')
        identifier = list(map(int, identifier_s.split(',')))

    if action == 'split':
        if dataset_name == 'adult':
            label='income_class'
        elif dataset_name == 'credit':
            label='kredit'
        elif dataset_name == 'compas':
            label='two_year_recid'
        else:
            ValueError("Dataset not recognized")

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
