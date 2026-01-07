import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances, roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KernelDensity

class DistanceBasedMembershipInference:
    def __init__(self, training_data: pd.DataFrame, test_data: pd.DataFrame, synthetic_data: pd.DataFrame, sample_size: int):
        self.training_data = training_data
        self.test_data = test_data
        self.sample_size = sample_size
        self.synthetic_data = synthetic_data
    
    def perform_inference(self):
        X_train = self.training_data
        X_test = self.test_data
        X_synthetic = self.synthetic_data

        scaler = MinMaxScaler()
        X_synthetic = scaler.fit_transform(X_synthetic)
        X_test = scaler.transform(X_test)
        X_train = scaler.transform(X_train)
        
        X_train = X_train[:self.sample_size]
        X_test = X_test[:self.sample_size]

        # Step 1: Compute minimum distance for each training sample from synthetic samples
        train_distances = pairwise_distances(X_train, X_synthetic)
        min_train_distances = np.min(train_distances, axis=1)
        
        # Step 2: Compute minimum distance for each test sample from synthetic samples
        test_distances = pairwise_distances(X_test, X_synthetic)
        min_test_distances = np.min(test_distances, axis=1)
        
        # Step 3: Combine distances and create labels
        # Create labels for the training and test data
        y_true = np.concatenate([np.ones(len(min_train_distances)), np.zeros(len(min_test_distances))])
        distances = np.concatenate([-min_train_distances, -min_test_distances])

        # Calculate AUC-ROC
        auc_roc = roc_auc_score(y_true, distances)

        return {
            'AUC-ROC': auc_roc,
            'Minimum Train Distances': min_train_distances,
            'Minimum Test Distances': min_test_distances
        }
    
class DistributionBasedMembershipInference:
    def __init__(self, training_data: pd.DataFrame, test_data: pd.DataFrame, synthetic_data: pd.DataFrame, sample_size: int):
        self.training_data = training_data
        self.test_data = test_data
        self.synthetic_data = synthetic_data
        self.sample_size = sample_size

    def perform_inference(self):
        # Separate features and labels
        X_train = self.training_data
        X_test = self.test_data
        X_synthetic = self.synthetic_data

        X_train = X_train[:self.sample_size]
        X_test = X_test[:self.sample_size]

        scaler = StandardScaler()
        X_synthetic = scaler.fit_transform(X_synthetic)
        X_test = scaler.transform(X_test)
        X_train = scaler.transform(X_train)

        pca = PCA(random_state=42, n_components=0.9).fit(X_synthetic)
        
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)
        X_synthetic = pca.transform(X_synthetic)

        kde = KernelDensity(kernel='exponential', bandwidth=0.05).fit(X_synthetic)

        log_density_train = kde.score_samples(X_train)
        log_density_test = kde.score_samples(X_test)

        log_density_combined = np.concatenate([log_density_train, log_density_test])

        y_true = np.concatenate([np.ones(len(log_density_train)), np.zeros(len(log_density_test))])

        # Calculate AUC-ROC and accuracy
        auc_roc = roc_auc_score(y_true, log_density_combined)

        return {
            'AUC-ROC': auc_roc,
            'Log Density Train': log_density_train,
            'Log Density Test': log_density_test
        }

class MonteCarloMembershipInference:
    def __init__(self, training_data: pd.DataFrame, test_data: pd.DataFrame, synthetic_data: pd.DataFrame, sample_size: int):
        self.training_data = training_data
        self.test_data = test_data
        self.synthetic_data = synthetic_data
        self.sample_size = sample_size
    
    def perform_inference(self):
        X_train = self.training_data
        X_test = self.test_data
        X_synthetic = self.synthetic_data

        scaler = MinMaxScaler()
        X_synthetic = scaler.fit_transform(X_synthetic)
        X_test = scaler.transform(X_test)
        X_train = scaler.transform(X_train)

        X_train = X_train[:self.sample_size]
        X_test = X_test[:self.sample_size]

        # Step 1: using the median heuristic: epsilon is the median between the minimum distances for all samples
        train_distances = np.min(pairwise_distances(X_train, X_synthetic), axis=1)
        test_distances = np.min(pairwise_distances(X_test, X_synthetic), axis=1)
        distances = np.concatenate([train_distances, test_distances])
        epsilon = np.median(distances)

        # Step 1: Compute minimum distance for each training sample from synthetic samples
        train_distances = pairwise_distances(X_train, X_synthetic)
        test_distances = pairwise_distances(X_test, X_synthetic)

        #for the estimation we will consider only sample close enough 
        mask_train = train_distances <= epsilon
        mask_test = test_distances <= epsilon

        log_distances_train = np.log(train_distances + 1e-10)
        log_distances_test = np.log(test_distances + 1e-10)

        density_estimate_train = -np.mean(mask_train * log_distances_train, axis=1)
        density_estimate_test = -np.mean(mask_test * log_distances_test, axis=1)

        log_density_combined = np.concatenate([density_estimate_train, density_estimate_test])

        y_true = np.concatenate([np.ones(len(density_estimate_train)), np.zeros(len(density_estimate_test))])

        # Calculate AUC-ROC and accuracy
        auc_roc = roc_auc_score(y_true, log_density_combined)

        return {
            'AUC-ROC': auc_roc,
            'Log Density Train': density_estimate_train,
            'Log Density Test': density_estimate_test
        }
    
class DOMIAS:
    def __init__(self, training_data: pd.DataFrame, test_data: pd.DataFrame, synthetic_data: pd.DataFrame, reference_size : int, sample_size: int):
        self.training_data = training_data
        self.test_data = test_data
        self.synthetic_data = synthetic_data
        self.sample_size = sample_size
        self.ref_size = reference_size
    
    def perform_inference(self):
        X_train = self.training_data
        X_test = self.test_data
        X_synthetic = self.synthetic_data
        X_ref = X_train[len(X_train)-self.ref_size:len(X_train)]
        X_train = X_train[:self.sample_size]
        X_test = X_test[:self.sample_size]

        scaler = StandardScaler()
        X_synthetic = scaler.fit_transform(X_synthetic)
        X_test = scaler.transform(X_test)
        X_train = scaler.transform(X_train)
        X_ref = scaler.transform(X_ref)

        pca = PCA(random_state=42, n_components=0.99).fit(X_synthetic)
        
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)
        X_synthetic = pca.transform(X_synthetic)
        X_ref = pca.transform(X_ref)

        # Step 1: Compute minimum distance for each training sample from synthetic samples
        kde_synth = KernelDensity(kernel='exponential', bandwidth=0.05).fit(X_synthetic)
        kde_ref = KernelDensity(kernel='exponential', bandwidth=0.15).fit(X_ref)

        log_density_train_synth = kde_synth.score_samples(X_train)
        log_density_test_synth = kde_synth.score_samples(X_test)
        log_density_train_ref = kde_ref.score_samples(X_train)
        log_density_test_ref = kde_ref.score_samples(X_test)

        log_density_train = log_density_train_synth / (log_density_train_ref + 1e-20)
        log_density_test = log_density_test_synth / (log_density_test_ref + 1e-20)
        
        log_density_combined = np.concatenate([-log_density_train, -log_density_test])
        y_true = np.concatenate([np.ones(len(log_density_train)), np.zeros(len(log_density_test))])

        # Calculate AUC-ROC and accuracy
        auc_roc = roc_auc_score(y_true, log_density_combined)

        return {
            'AUC-ROC': auc_roc,
            'Log Density Train': log_density_train,
            'Log Density Test': log_density_test
        }

class PrivacyMetricsEvaluator:
    def __init__(self, training_data: pd.DataFrame, test_data: pd.DataFrame, synthetic_data: pd.DataFrame):
        self.scaler = MinMaxScaler().fit(training_data) #cant use synthetic data otherwise the comparison is more difficult

        self.training_data = self.scaler.transform(training_data)
        self.test_data = self.scaler.transform(test_data)
        self.synthetic_data = self.scaler.transform(synthetic_data)

        # Computing DCR considering only the training data is removing the context on which the data was generated
        # Considering both training and test data (holdout) has the chance of taking into account the distribution of the data
        # If the data is concentrated in one region, the DCR will be lower for both synthetic and real data

        self.distances_st = pairwise_distances(self.synthetic_data, self.training_data)
        self.distances_tt = pairwise_distances(self.synthetic_data, self.test_data)
        self.distances_train = pairwise_distances(self.training_data, self.training_data)
        np.fill_diagonal(self.distances_train, np.inf)
        self.distances_test = pairwise_distances(self.test_data, self.test_data)
        np.fill_diagonal(self.distances_test, np.inf)
        self.distances_synth = pairwise_distances(self.synthetic_data, self.synthetic_data)
        np.fill_diagonal(self.distances_synth, np.inf)

    def compute_dcr(self):
        """Compute the Distance to Closest Record (DCR) for each synthetic record."""
        
        dcr_scores_train = np.min(self.distances_st, axis=1)
        dcr_scores_test = np.min(self.distances_tt, axis=1)

        #for each synthetic record, check if the closest record is in the train or in the test set
        dcr = (dcr_scores_train - dcr_scores_test) < 0 #if true, the closest record is in the training set
        
        return sum(dcr)/len(dcr)

    def compute_nndr(self):
        """Compute the Nearest Neighbour Distance Ratio (NNDR) for each synthetic record."""
        
        nndr_scores_train = []
        for dist in self.distances_st:
            # Sort distances to find the nearest and second nearest neighbors
            sorted_distances = np.sort(dist)
            nearest_distance = sorted_distances[0]
            second_nearest_distance = sorted_distances[1]
            nndr = nearest_distance / second_nearest_distance
            nndr_scores_train.append(nndr)
        nndr_scores_train = np.array(nndr_scores_train)

        nndr_scores_test = []
        for dist in self.distances_tt:
            # Sort distances to find the nearest and second nearest neighbors
            sorted_distances = np.sort(dist)
            nearest_distance = sorted_distances[0]
            second_nearest_distance = sorted_distances[1]
            nndr = nearest_distance / second_nearest_distance
            nndr_scores_test.append(nndr)
        nndr_scores_test = np.array(nndr_scores_test)

        nndr = (nndr_scores_train - nndr_scores_test) < 0
        return sum(nndr)/len(nndr)
    
    def compute_adversarial_accuracy(self, D_TT, D_ST, D_SS):    
        # Compute minimum distances
        d_SS = np.min(D_SS, axis=1)
        d_TT = np.min(D_TT, axis=1)
        d_TS = np.min(D_ST, axis=0)
        d_ST = np.min(D_ST, axis=1)
        
        term1 = np.mean(d_TS > d_TT)
        term2 = np.mean(d_ST > d_SS)
        
        # Compute adversarial accuracy
        return 0.5 * (term1 + term2)

    def compute_privacy_loss(self):
        AA_train = self.compute_adversarial_accuracy(self.distances_train, self.distances_st, self.distances_synth)
        AA_test = self.compute_adversarial_accuracy(self.distances_test, self.distances_tt, self.distances_synth)
        return AA_test - AA_train

    def compute_privacy_metrics(self):
        """Compute a set of privacy metrics."""
        return {
            'DCR_Synth': self.compute_dcr(),
            'NNDR_Synth': self.compute_nndr(),
            'Loss': self.compute_privacy_loss()
        }