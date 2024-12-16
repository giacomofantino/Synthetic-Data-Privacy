import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from scipy.stats import wasserstein_distance
from scipy.spatial import distance
import numpy as np
from dython.nominal import associations
import xgboost as xgb

class UtilityEvaluator:
    def __init__(self, training_data: pd.DataFrame, test_data: pd.DataFrame, synthetic_data: pd.DataFrame):
        self.training_data = training_data
        self.test_data = test_data
        self.synthetic_data = synthetic_data

    def evaluate(self, categorical_columns=None):
        label='income_class'
        # Separate features and labels
        X_train = self.training_data.drop(columns=label)
        y_train = self.training_data[label]
        X_test = self.test_data.drop(columns=label)
        y_test = self.test_data[label]
        X_synthetic = self.synthetic_data.drop(columns=label)
        y_synthetic = self.synthetic_data[label]

        real_corr = associations(self.training_data, nominal_columns=categorical_columns, compute_only=True)['corr']

        fake_corr = associations(self.synthetic_data, nominal_columns=categorical_columns, compute_only=True)['corr']

        corr_dist = np.linalg.norm(real_corr - fake_corr)
    
        jsd_categorical = []
        wd_numerical = []
        for col in self.training_data.columns:
            if col in categorical_columns:
                p = self.training_data[col].value_counts(normalize=True)
                q = self.synthetic_data[col].value_counts(normalize=True)
                all_categories = set(p.index).union(q.index)
                p = p.reindex(all_categories, fill_value=0).values
                q = q.reindex(all_categories, fill_value=0).values
                jsd_categorical.append(distance.jensenshannon(p, q))
            else:
                scaler = MinMaxScaler()
                scaler.fit(self.training_data[col].values.reshape(-1,1))
                l1 = scaler.transform(self.training_data[col].values.reshape(-1,1)).flatten()
                l2 = scaler.transform(self.synthetic_data[col].values.reshape(-1,1)).flatten()
                wd_numerical.append(wasserstein_distance(l1, l2))


            results = {}
            results = {
                'JSD': np.mean(jsd_categorical),
                'WSD': np.mean(wd_numerical),
                'corr-dist': corr_dist
            }

        model_train = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        model_train.fit(X_train, y_train)
        
        model_synthetic = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        model_synthetic.fit(X_synthetic, y_synthetic)

        # Use train data
        y_pred_train = model_train.predict(X_test)
        y_pred_synthetic = model_synthetic.predict(X_test)
        
        #return the difference between using synthetic and train data
        return results, accuracy_score(y_test, y_pred_synthetic) - accuracy_score(y_test, y_pred_train)