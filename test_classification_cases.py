import numpy as np
import pandas as pd
from unittest import TestCase
from unittest.mock import Mock, patch
from classification_cases import ClassificationCases
from sklearn.model_selection import GridSearchCV
import warnings

class TestClassificationCases(TestCase):
    def setUp(self):
        # Generate a synthetic dataset
        self.df = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'target': np.random.randint(0, 2, size=100)  # Binary classification target
        })
        self.target_col = 'target'
        self.classification_cases = ClassificationCases(self.df, self.target_col)

    def test_constructor(self):
        # Check if the instance created in setUp is of the correct type and if attributes are correctly initialized
        self.assertTrue(isinstance(self.classification_cases, ClassificationCases))
        self.assertEqual(self.classification_cases.df.shape, self.df.shape)  # Check if the DataFrame shape matches
        self.assertEqual(self.classification_cases.target_col, self.target_col)  # Check if the target column matches

    def test_split_data(self):
        # Call the split_data method
        self.classification_cases.split_data()

        # Check if the split data attributes are not None
        self.assertIsNotNone(self.classification_cases.X_train)
        self.assertIsNotNone(self.classification_cases.X_test)
        self.assertIsNotNone(self.classification_cases.y_train)
        self.assertIsNotNone(self.classification_cases.y_test)

    def test_scale_data(self):
        self.classification_cases.X_train = self.df[['feature1', 'feature2']]
        self.classification_cases.X_test = self.df[['feature1', 'feature2']]

        # Call the scale_data method to get the scaled data
        scaled_X_train, scaled_X_test = self.classification_cases.scale_data()

        # Check if the mean of scaled data is close to 0
        self.assertTrue(np.allclose(np.mean(scaled_X_train, axis=0), np.zeros(scaled_X_train.shape[1])))
        self.assertTrue(np.allclose(np.mean(scaled_X_test, axis=0), np.zeros(scaled_X_test.shape[1])))

        # Check if the standard deviation of scaled data is close to 1
        self.assertTrue(np.allclose(np.std(scaled_X_train, axis=0), np.ones(scaled_X_train.shape[1])))
        self.assertTrue(np.allclose(np.std(scaled_X_test, axis=0), np.ones(scaled_X_test.shape[1])))

    def test_train_models(self):
        # Call the split_data method to create X_train and X_test
        self.classification_cases.split_data()

        # Call the scale_data method to create scaled_X_train and scaled_X_test
        scaled_X_train, scaled_X_test = self.classification_cases.scale_data()

        # Call the train_models method after scaling the data
        self.classification_cases.train_models()

        # Now, you can check if the models have been trained
        self.assertIsNotNone(self.classification_cases.log_model)
        self.assertIsNotNone(self.classification_cases.knn_model)
        self.assertIsNotNone(self.classification_cases.svc_model)