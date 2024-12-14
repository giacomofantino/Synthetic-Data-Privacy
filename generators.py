import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import Metadata
from ctgan import CTGAN
#from synthcity.plugins import Plugins

rng = np.random.default_rng(42)

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
    def __init__(self, epochs=100, discrete_columns=[]):
        super().__init__()
        self.epochs = epochs
        self.discrete_columns = discrete_columns
        self.ctgan = CTGAN(verbose=True)
        self.ctgan.set_random_state(42)

    def train(self, training_data: pd.DataFrame):
        """Train the CTGAN on the provided training data."""
        self.ctgan.fit(training_data, self.discrete_columns, epochs=self.epochs)

    def generate(self, num_samples: int) -> pd.DataFrame:
        """Generate synthetic data samples using the trained CTGAN."""
        return self.ctgan.sample(num_samples)

class PATEGANDataGenerator(TabularDataGenerator):
    def __init__(self, epochs=100):
        super().__init__()
        self.epochs = epochs
        self.pate_gan = Plugins().get("pategan", generator_n_iter=self.epochs)

    def train(self, training_data: pd.DataFrame):
        """Train PATE-GAN on the provided training data."""
        self.pate_gan.fit(training_data, max_iter=self.epochs)

    def generate(self, num_samples: int) -> pd.DataFrame:
        """Generate synthetic data samples using the trained PATE-GAN."""
        synthetic_data = self.pate_gan.generate(num_samples).dataframe()
        return synthetic_data

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

class SDVDataGenerator(TabularDataGenerator):
    def __init__(self):
        super().__init__()

    def train(self, training_data: pd.DataFrame):
        """Train the SDV model on the provided training data."""
        metadata = Metadata.detect_from_dataframe(data=training_data)
        self.model = GaussianCopulaSynthesizer(metadata=metadata, default_distribution='gaussian_kde')
        self.model.fit(training_data)

    def generate(self, num_samples: int) -> pd.DataFrame:
        """Generate synthetic data samples using the trained SDV model."""
        return self.model.sample(num_rows=num_samples)