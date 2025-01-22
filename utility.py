import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import xgboost as xgb

class UtilityEvaluator:
    def __init__(self, training_data: pd.DataFrame, test_data: pd.DataFrame, synthetic_data: pd.DataFrame, label):
        self.training_data = training_data
        self.test_data = test_data
        self.synthetic_data = synthetic_data
        self.label = label

    def evaluate(self):
        # Separate features and labels
        X_train = self.training_data.drop(columns=self.label)
        y_train = self.training_data[self.label]
        X_test = self.test_data.drop(columns=self.label)
        y_test = self.test_data[self.label]
        X_synthetic = self.synthetic_data.drop(columns=self.label)
        y_synthetic = self.synthetic_data[self.label]

        if y_train.nunique() > 2:
            binary = False
        else:
            binary = True
        
        model_train = xgb.XGBClassifier(eval_metric="logloss")
        model_train.fit(X_train, y_train)

        model_synthetic = xgb.XGBClassifier(eval_metric="logloss")
        model_synthetic.fit(X_synthetic, y_synthetic)
        
        # Use train data
        y_pred_train = model_train.predict_proba(X_test)
        y_pred_synthetic = model_synthetic.predict_proba(X_test)
        y_class_train = y_pred_train.argmax(axis=1)
        y_class_synthetic = y_pred_synthetic.argmax(axis=1)
        
        #return the difference between using synthetic and train data
        diff_accuracy = accuracy_score(y_test, y_class_synthetic) - accuracy_score(y_test, y_class_train)
        if binary:
            diff_f1 = f1_score(y_test, y_class_synthetic) - f1_score(y_test, y_class_train)
        else:
            diff_f1 = f1_score(y_test, y_class_synthetic, average="weighted") - f1_score(y_test, y_class_train, average="weighted")

        if binary:
            #for the binary case, the predict_proba returns only the probability of the positive class
            diff_auc = roc_auc_score(y_test, y_pred_synthetic[:,1]) - roc_auc_score(y_test, y_pred_train[:,1])
        else:
            diff_auc = roc_auc_score(y_test, y_pred_synthetic, average="weighted", multi_class="ovo") - roc_auc_score(y_test, y_pred_train, average="weighted", multi_class="ovo")

        return diff_accuracy, diff_f1, diff_auc