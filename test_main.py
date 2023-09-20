from unittest import TestCase
from unittest.mock import patch
import numpy as np
import pandas as pd
from main import ML_Wizard
from regression_cases import RegressionCases
from classification_cases import ClassificationCases



class Test_ML_Wizard(TestCase):
    def test_constructor(self):
        # Test the constructor to ensure the instance is created correctly
        ml_wizard = ML_Wizard()
        self.assertIsNone(ml_wizard.ml_type)
        self.assertIsNone(ml_wizard.df)
        self.assertIsNone(ml_wizard.target_col)


    def test_get_ml_type(self):
        # Test the get_ml_type method
        ml_wizard = ML_Wizard()
        # Simulate user input
        ml_wizard.ml_type = 'regressor'
        self.assertEqual(ml_wizard.ml_type, 'regressor')
        ml_wizard.ml_type = 'classifier'
        self.assertEqual(ml_wizard.ml_type, 'classifier')


    def test_read_csv_file(self):
        # Test the read_csv_file method
        ml_wizard = ML_Wizard()
        # Simulate user input
        ml_wizard.read_csv_file()
        # Check if df is not None after reading the file
        self.assertIsNotNone(ml_wizard.df)


    def test_get_target_column(self):
        # Test the get_target_column method
        ml_wizard = ML_Wizard()
        ml_wizard.df = pd.DataFrame({'col1': [1, 2, 3], 'target': [4, 5, 6]})
        # Simulate user input
        ml_wizard.get_target_column()
        self.assertEqual(ml_wizard.target_col, 'target')


    def test_preprocess_data(self):
        # Test when there are no missing values and no categorical columns
        ml_wizard = ML_Wizard()
        ml_wizard.df = pd.DataFrame({'col1': [1, 2, 3], 'target': [4, 5, 6]})
        ml_wizard.preprocess_data()
        self.assertEqual(len(ml_wizard.df.columns), 2)  # Ensure no columns were added or removed.

        # # Test when there are missing values
        ml_wizard = ML_Wizard()
        ml_wizard.df = pd.DataFrame({'col1': [1, 2, np.nan], 'target': [4, 5, 6]})
        with self.assertRaises(ValueError):
            ml_wizard.preprocess_data() # Expecting an exception for missing values.

        # Test when there are categorical columns and user chooses to convert
        ml_wizard = ML_Wizard()
        ml_wizard.df = pd.DataFrame({'col1': ['A', 'B', 'C'], 'target': [4, 5, 6]})
        ml_wizard.preprocess_data()
        self.assertTrue('col1_B' in ml_wizard.df.columns) # Ensure one-hot encoding was applied.

        # Test when there are categorical columns and user chooses not to convert
        ml_wizard = ML_Wizard()
        ml_wizard.df = pd.DataFrame({'col1': ['A', 'B', 'C'], 'target': [4, 5, 6]})
        with self.assertRaises(ValueError) as context:
            ml_wizard.preprocess_data()
        
        expected_error_message = ("Data is not ready for machine learning process. "
                                "There are categorical/string values in the data. "
                                "Convert them and rerun the app.")
        self.assertEqual(str(context.exception), expected_error_message)

        # Test when the user provides an invalid response
        ml_wizard = ML_Wizard()
        ml_wizard.df = pd.DataFrame({'col1': ['A', 'B', 'C'], 'target': [4, 5, 6]})
        with self.assertRaises(ValueError) as context:
            ml_wizard.preprocess_data()
        
        expected_error_message_part = "Invalid response. Please enter 'yes' or 'no'."
        actual_error_message = str(context.exception)
        self.assertEqual(expected_error_message_part, actual_error_message)


    def test_run_machine_learning(self):
        # Test when ml_type is 'regressor'
        ml_wizard = ML_Wizard()
        ml_wizard.ml_type = 'regressor'

        # Create a mock DataFrame with appropriate columns, including the target column
        mock_df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': [7, 8, 9] # Include the target column
        })
        ml_wizard.df = mock_df # Set the mock DataFrame
        ml_wizard.target_col = 'target' # Assign the name of the target column

        # Use patch to create mock objects for the methods of RegressionCases
        with patch.object(RegressionCases, 'train_models') as mock_train_models:
            with patch.object(RegressionCases, 'grid_search') as mock_grid_search:
                with patch.object(RegressionCases, 'evaluate_models') as mock_evaluate_models:
                    ml_wizard.run_machine_learning()

        # Check if the methods of RegressionCases are called as expected
        mock_train_models.assert_called_once()
        mock_grid_search.assert_called_once()
        mock_evaluate_models.assert_called_once()

        # Test when ml_type is 'classifier'
        ml_wizard = ML_Wizard()
        ml_wizard.ml_type = 'classifier'

        # Create a mock DataFrame with appropriate columns, including the target column
        mock_df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'feature2': [10, 11, 12, 13, 14, 15, 16, 17, 18],
            'target': [0, 1, 0, 0, 1, 0, 1, 1, 1] # Include the target column with binary classification labels
        })
        ml_wizard.df = mock_df # Set the mock DataFrame
        ml_wizard.target_col = 'target' # Assign the name of the target column

        # Use patch to create mock objects for the methods of ClassificationCases
        with patch.object(ClassificationCases, 'split_data') as mock_split_data:
            with patch.object(ClassificationCases, 'scale_data') as mock_scale_data:
                with patch.object(ClassificationCases, 'train_models') as mock_train_models:
                    with patch.object(ClassificationCases, 'grid_search') as mock_grid_search:
                        with patch.object(ClassificationCases, 'evaluate_models') as mock_evaluate_models:
                            ml_wizard.run_machine_learning()

        # Check if the methods of ClassificationCases are called as expected
        mock_split_data.assert_called_once()
        mock_scale_data.assert_called_once()
        mock_train_models.assert_called_once()
        mock_grid_search.assert_called_once()
        mock_evaluate_models.assert_called_once()