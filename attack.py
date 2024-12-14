import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity

class DistanceBasedMembershipInference:
    def __init__(self, training_data: pd.DataFrame, test_data: pd.DataFrame, synthetic_data: pd.DataFrame):
        self.training_data = training_data
        self.test_data = test_data
        self.synthetic_data = synthetic_data
    
    def perform_inference(self, scale_data=True):
        X_train = self.training_data
        X_test = self.test_data
        X_synthetic = self.synthetic_data
        
        X_train = X_train[:len(X_test)]

        if scale_data:
            scaler = StandardScaler()
            X_synthetic = scaler.fit_transform(X_synthetic)
            X_test = scaler.transform(X_test)
            X_train = scaler.transform(X_train)
        
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
    def __init__(self, training_data: pd.DataFrame, test_data: pd.DataFrame, synthetic_data: pd.DataFrame):
        self.training_data = training_data
        self.test_data = test_data
        self.synthetic_data = synthetic_data

    def perform_inference(self, scale_data=False):
        # Separate features and labels
        X_train = self.training_data
        X_test = self.test_data
        X_synthetic = self.synthetic_data
        
        X_train = X_train[:len(X_test)]

        if scale_data: ##check se ha senso o meno
            scaler = StandardScaler()
            X_synthetic = scaler.fit_transform(X_synthetic)
            X_test = scaler.transform(X_test)
            X_train = scaler.transform(X_train)

        kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X_synthetic)

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
    def __init__(self, training_data: pd.DataFrame, test_data: pd.DataFrame, synthetic_data: pd.DataFrame):
        self.training_data = training_data
        self.test_data = test_data
        self.synthetic_data = synthetic_data
    
    def perform_inference(self, scale_data=True):
        X_train = self.training_data
        X_test = self.test_data
        X_synthetic = self.synthetic_data
        
        X_train = X_train[:len(X_test)]

        if scale_data:
            scaler = StandardScaler()
            X_synthetic = scaler.fit_transform(X_synthetic)
            X_test = scaler.transform(X_test)
            X_train = scaler.transform(X_train)

        # Step 1: using the median heuristic: epsilon is the median between the minimum distances for all samples
        train_distances = np.min(pairwise_distances(X_train, X_synthetic), axis=1)
        test_distances = np.min(pairwise_distances(X_test, X_synthetic), axis=1)
        distances = np.concatenate([train_distances, test_distances])
        epsilon = np.median(distances)

        # Step 1: Compute minimum distance for each training sample from synthetic samples
        train_distances = pairwise_distances(X_train, X_synthetic)
        test_distances = pairwise_distances(X_test, X_synthetic)

        mask_train = train_distances <= epsilon #if value false is equal to zero
        mask_test = test_distances <= epsilon #if value false is equal to zero

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
    def __init__(self, training_data: pd.DataFrame, test_data: pd.DataFrame, synthetic_data: pd.DataFrame, reference_data : pd.DataFrame):
        self.training_data = training_data
        self.test_data = test_data
        self.synthetic_data = synthetic_data
        self.ref_data = reference_data
    
    def perform_inference(self, scale_data=False):
        X_train = self.training_data
        X_test = self.test_data
        X_synthetic = self.synthetic_data
        X_ref = self.ref_data
        
        X_train = X_train[:len(X_test)]

        if scale_data:
            scaler = StandardScaler()
            X_synthetic = scaler.fit_transform(X_synthetic)
            X_test = scaler.transform(X_test)
            X_train = scaler.transform(X_train)

        # Step 1: Compute minimum distance for each training sample from synthetic samples
        kde_synth = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X_synthetic)
        kde_ref = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X_ref)

        log_density_train_synth = kde_synth.score_samples(X_train)
        log_density_test_synth = kde_synth.score_samples(X_test)
        log_density_train_ref = kde_ref.score_samples(X_train)
        log_density_test_ref = kde_ref.score_samples(X_test)

        log_density_train = log_density_train_synth / (log_density_train_ref + 1e-10)
        log_density_test = log_density_test_synth / (log_density_test_ref + 1e-10)
        
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
        self.training_data = training_data
        self.test_data = test_data
        self.synthetic_data = synthetic_data
        # Compute pairwise distances between synthetic data and training data
        self.distances_st = pairwise_distances(self.synthetic_data, self.training_data)
        self.distances_tt = pairwise_distances(self.test_data, self.training_data)

    def compute_dcr(self):
        """Compute the Distance to Closest Record (DCR) for each synthetic record."""
        # DCR is the minimum distance to the closest training record for each synthetic record
        dcr_scores_synth = np.min(self.distances_st, axis=1)
        dcr_scores_test = np.min(self.distances_tt, axis=1)
        return dcr_scores_synth, dcr_scores_test

    def compute_nndr(self):
        """Compute the Nearest Neighbour Distance Ratio (NNDR) for each synthetic record."""
        nndr_scores_synth = []
        for dist in self.distances_st:
            # Sort distances to find the nearest and second nearest neighbors
            sorted_distances = np.sort(dist)
            nearest_distance = sorted_distances[0]
            second_nearest_distance = sorted_distances[1]
            nndr = nearest_distance / second_nearest_distance
            nndr_scores_synth.append(nndr)
        
        nndr_scores_test = []
        for dist in self.distances_tt:
            # Sort distances to find the nearest and second nearest neighbors
            sorted_distances = np.sort(dist)
            nearest_distance = sorted_distances[0]
            second_nearest_distance = sorted_distances[1]
            nndr = nearest_distance / second_nearest_distance
            nndr_scores_test.append(nndr)

        return np.array(nndr_scores_synth), np.array(nndr_scores_test)

    def compute_privacy_metrics(self):
        """Compute a set of privacy metrics."""
        dcr_scores_synth, dcr_scores_test = self.compute_dcr()
        nndr_scores_synth, nndr_scores_test = self.compute_nndr()

        return {
            'DCR_Synth': dcr_scores_synth,
            'DCR_Test': dcr_scores_test,
            'NNDR_Synth': nndr_scores_synth,
            'NNDR_Test': nndr_scores_test,
        }
