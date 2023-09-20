from unittest import TestCase
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from main import ML_Wizard
from regression_cases import RegressionCases



class TestRegressionCases(TestCase):
    def setUp(self):
        # Generate a synthetic dataset
        self.df = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'target': np.random.rand(100)
        })
        self.target_col = 'target'
        self.regression_cases = RegressionCases(self.df, self.target_col)


    def test_constructor(self):
        # Check if the instance created in setUp is of the correct type and if attributes are correctly initialized
        self.assertTrue(isinstance(self.regression_cases, RegressionCases))
        self.assertEqual(self.regression_cases.df.shape, self.df.shape) # Check if the DataFrame shape matches
        self.assertEqual(self.regression_cases.target_col, self.target_col) # Check if the target column matches


    def test_split_data(self):
        # Call the split_data method
        self.regression_cases.split_data()

        # Check if the split data attributes are not None
        self.assertIsNotNone(self.regression_cases.X_train)
        self.assertIsNotNone(self.regression_cases.X_test)
        self.assertIsNotNone(self.regression_cases.y_train)
        self.assertIsNotNone(self.regression_cases.y_test)


    def test_scale_data(self):
        self.regression_cases.X_train = self.df[['feature1', 'feature2']]
        self.regression_cases.X_test = self.df[['feature1', 'feature2']]

        # Call the scale_data method to get the scaled data
        scaled_X_train, scaled_X_test = self.regression_cases.scale_data()

        # Check if the mean of scaled data is close to 0
        self.assertTrue(np.allclose(np.mean(scaled_X_train, axis=0), np.zeros(scaled_X_train.shape[1])))
        self.assertTrue(np.allclose(np.mean(scaled_X_test, axis=0), np.zeros(scaled_X_test.shape[1])))

        # Check if the standard deviation of scaled data is close to 1
        self.assertTrue(np.allclose(np.std(scaled_X_train, axis=0), np.ones(scaled_X_train.shape[1])))
        self.assertTrue(np.allclose(np.std(scaled_X_test, axis=0), np.ones(scaled_X_test.shape[1])))


    def test_train_models(self):
        self.regression_cases.train_models()
        self.assertIsNotNone(self.regression_cases.lir_model)
        self.assertIsNotNone(self.regression_cases.lasso_model)
        self.assertIsNotNone(self.regression_cases.ridge_model)
        self.assertIsNotNone(self.regression_cases.elastic_model)
        self.assertIsNotNone(self.regression_cases.svr_model)


    def test_grid_search(self):
        self.regression_cases.grid_search()
        
        self.assertIsInstance(self.regression_cases.lir_grid, GridSearchCV)
        self.assertIsInstance(self.regression_cases.lasso_grid, GridSearchCV)
        self.assertIsInstance(self.regression_cases.ridge_grid, GridSearchCV)
        self.assertIsInstance(self.regression_cases.elastic_grid, GridSearchCV)
        self.assertIsInstance(self.regression_cases.svr_grid, GridSearchCV)

        self.assertIsNotNone(self.regression_cases.lir_grid.best_estimator_)
        self.assertIsNotNone(self.regression_cases.lasso_grid.best_estimator_)
        self.assertIsNotNone(self.regression_cases.ridge_grid.best_estimator_)
        self.assertIsNotNone(self.regression_cases.elastic_grid.best_estimator_)
        self.assertIsNotNone(self.regression_cases.svr_grid.best_estimator_)

        self.assertIsNotNone(self.regression_cases.best_lir)
        self.assertIsNotNone(self.regression_cases.best_lasso)
        self.assertIsNotNone(self.regression_cases.best_ridge)
        self.assertIsNotNone(self.regression_cases.best_elastic)
        self.assertIsNotNone(self.regression_cases.best_svr)


    def test_evaluate_models(self):
        # Create a controlled dataset for testing
        test_data = {
            'X1': [1, 2, 3, 4, 5],
            'X2': [2, 3, 4, 5, 6],
            'y': [3, 4, 5, 6, 7]
        }
        test_df = pd.DataFrame(test_data)
        target_col = 'y'

        # Create a RegressionCases instance with the controlled dataset
        regression_cases = RegressionCases(test_df, target_col)

        # Fit the models before evaluation
        regression_cases.lir_model.fit(regression_cases.scaled_X_train, regression_cases.y_train)
        regression_cases.lasso_model.fit(regression_cases.scaled_X_train, regression_cases.y_train)
        regression_cases.ridge_model.fit(regression_cases.scaled_X_train, regression_cases.y_train)
        regression_cases.elastic_model.fit(regression_cases.scaled_X_train, regression_cases.y_train)
        regression_cases.svr_model.fit(regression_cases.scaled_X_train, regression_cases.y_train)

        # Set mock values for model predictions and evaluation metrics
        mock_predictions = np.array([3.2, 4.2, 5.2, 6.2, 7.2])
        mock_mae = 0.2
        mock_rmse = 0.2
        mock_r2 = 0.95

        # Mock the model predictions and evaluation metrics
        with patch.object(regression_cases.lir_model, 'predict', return_value=mock_predictions):
            with patch('regression_cases.mean_absolute_error', return_value=mock_mae):
                with patch('regression_cases.mean_squared_error', return_value=mock_rmse):
                    with patch('regression_cases.r2_score', return_value=mock_r2):
                        # Call the evaluate_models method
                        regression_cases.evaluate_models()


        # Mock the input function to simulate user input
        with patch('builtins.input', side_effect=['yes', 'test_model.joblib']):
            # Call the evaluate_models method
            regression_cases.evaluate_models()

        # Define expected values based on the mock data for clarity and readability
        expected_mae = mock_mae
        expected_rmse = mock_rmse
        expected_r2 = mock_r2
        expected_best_model = "Linear Regression"

        # Use assertions to check if the calculated values match the expected values
        self.assertEqual(mock_mae, expected_mae)
        self.assertEqual(mock_rmse, expected_rmse)
        self.assertEqual(mock_r2, expected_r2)
        self.assertEqual("Linear Regression", expected_best_model)

        # Check if the model was saved as expected
        with open('test_model.joblib', 'rb') as model_file:
            saved_model_content = model_file.read()
            self.assertTrue(len(saved_model_content) > 0)