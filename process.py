from sklearn.preprocessing import LabelEncoder
import pickle
import os

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}

    def fit_transform(self, dataset):
        """
        Fit LabelEncoders to the categorical columns of the dataset and transform the dataset.

        Parameters:
        dataset (pd.DataFrame): The input dataset to preprocess.

        Returns:
        pd.DataFrame: The transformed dataset with numerical values.
        dict: A dictionary containing the label encodings for each categorical column.
        """
        # Create a copy of the dataset to avoid modifying the original
        transformed_dataset = dataset.copy()

        # Iterate through each column in the dataset
        for column in transformed_dataset.select_dtypes(include=['object']).columns:
            # Initialize a LabelEncoder for the column
            le = LabelEncoder()
            # Fit the encoder and transform the column
            transformed_dataset[column] = le.fit_transform(transformed_dataset[column])
            # Store the encoder in the dictionary
            self.label_encoders[column] = le

        return transformed_dataset
    
    def save_label_encoders(self, directory_path, filename='label_encoders.pkl'):
        """
        Save the label encoders to a specified directory.

        Parameters:
        directory (str): The directory where the label encoders will be saved.
        filename (str): The name of the file to save the label encoders as.
        """

        file_path = os.path.join(directory_path, filename)
        # Save the label encoders using pickle
        with open(file_path, 'wb') as file:
            pickle.dump(self.label_encoders, file)

