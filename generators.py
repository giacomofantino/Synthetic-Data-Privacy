import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from ctgan.synthesizers import TVAE
from ctgan import CTGAN
import torch
import os
import random
from CTABGAN.model.ctabgan import CTABGAN

#replicate results of GANs
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.use_deterministic_algorithms(True)
np.random.seed(42)
random.seed(42)
rng = np.random.default_rng(42)

# Additional settings for deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class TabularDataGenerator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def train(self, training_data: pd.DataFrame):
        """Train the generator on the provided training data."""
        pass

    @abstractmethod
    def generate(self, num_samples: int) -> pd.DataFrame:
        """Generate synthetic data samples."""
        pass

class CTGANDataGenerator(TabularDataGenerator):
    def __init__(self, epochs=100, discrete_columns=[], batch_size=500, discriminator_steps=1):
        super().__init__()
        self.epochs = epochs
        self.discrete_columns = discrete_columns
        self.ctgan = CTGAN(verbose=True, batch_size=batch_size, discriminator_steps=discriminator_steps)
        self.ctgan.set_random_state(42)

    def train(self, training_data: pd.DataFrame):
        """Train the CTGAN on the provided training data."""
        self.ctgan.fit(training_data, self.discrete_columns, epochs=self.epochs)

    def generate(self, num_samples: int) -> pd.DataFrame:
        """Generate synthetic data samples using the trained CTGAN."""
        return self.ctgan.sample(num_samples)

class MixupDataGenerator(TabularDataGenerator):
    def __init__(self, integer_columns, alpha=0.2):
        super().__init__()
        self.alpha = alpha
        self.integer_columns = integer_columns

    def train(self, training_data: pd.DataFrame):
        """Store the training data for generating mixup samples."""
        self.training_data = training_data

    def generate(self, num_samples: int) -> pd.DataFrame:
        """Generate synthetic data samples using the Mixup technique."""
        num_rows = len(self.training_data)
        synthetic_data = []

        for _ in range(num_samples):
            # Randomly select two samples
            idx1, idx2 = rng.choice(num_rows, 2, replace=False)
            sample1 = self.training_data.iloc[idx1].values
            sample2 = self.training_data.iloc[idx2].values
            
            # Generate a random lambda value from a beta distribution
            lambda_value = rng.beta(self.alpha, self.alpha)
            
            # Create a new sample as a mix of the two samples
            mixed_sample = lambda_value * sample1 + (1 - lambda_value) * sample2
            synthetic_data.append(mixed_sample)
        
        synthetic_data = pd.DataFrame(synthetic_data, columns=self.training_data.columns)

        #columns that were not continuous
        for col in self.integer_columns:
            synthetic_data[col] = synthetic_data[col].round().astype(int)

        return synthetic_data

class TVAEDataGenerator(TabularDataGenerator):
    def __init__(self, epochs=100, discrete_columns=[], batch_size=500):
        super().__init__()
        self.epochs=epochs
        self.discrete_columns = discrete_columns
        self.model = TVAE(epochs=self.epochs, verbose=True, batch_size=batch_size)
        self.model.set_random_state(42)

    def train(self, training_data: pd.DataFrame):
        """Train the TVAE model on the provided training data."""
        self.model.fit(training_data, self.discrete_columns)

    def generate(self, num_samples: int) -> pd.DataFrame:
        """Generate synthetic data samples using the trained TVAE model."""
        return self.model.sample(num_samples)

class CTABGANDataGenerator(TabularDataGenerator):
    def __init__(self, train_file_path,epochs=100, batch_size=500,integer_columns=[], categorical_columns=[], mixed_columns={}, 
                 general_columns=[], problem_type={}, log_columns=[], discriminator_steps=1):
        super().__init__()
        integer_columns = list(set(integer_columns) - set(categorical_columns))
        self.ctabgan = CTABGAN(raw_csv_path=train_file_path, integer_columns=integer_columns, categorical_columns=categorical_columns,
                               mixed_columns= mixed_columns, general_columns = general_columns, epochs=epochs, problem_type=problem_type, batch_size=batch_size,
                               log_columns=log_columns)
        self.epochs = epochs
        self.discriminator_steps = discriminator_steps
        
    def train(self, training_data: pd.DataFrame):
        self.ctabgan.fit(self.discriminator_steps)
    
    def generate(self, num_samples: int) -> pd.DataFrame:
        return self.ctabgan.generate_samples(num_samples)