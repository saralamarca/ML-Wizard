import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import seaborn as sns
from classification_cases import ClassificationCases
from regression_cases import RegressionCases



class ML_Wizard:
    """
    A class for building and running supervised machine learning models.
    """
    def __init__(self):
        """
        Initialize the ML_Wizard instance.
        """
        self.ml_type = None # Will store the type of machine learning model (regressor or classifier).
        self.df = None # Will store the dataset.
        self.target_col = None # Will store the name of the dependent target column.

    def get_ml_type(self):
        """
        Prompt the user to select the type of machine learning model (regressor or classifier).

        Returns:
            None
        """
        # Read from the user what type of machine learning model they need.
        while True:
            self.ml_type = input("What kind of supervised machine learning model do you need? "
                                 "Enter 'regressor' or 'classifier': ")
            if self.ml_type not in ['regressor', 'classifier']:
                print("Invalid input.")
            else:
                break

    def read_csv_file(self):
        """
        Prompt the user to enter the path or filename of the CSV file containing the data.

        Returns:
            None
        """
        # Read the dataset from the CSV file.
        while True:
            file_name = input("Enter the path or .csv file name (format example xxx.csv): ")
            try:
                self.df = pd.read_csv(file_name)
                break
            except FileNotFoundError:
                print(f"File not found at {file_name}. "
                      "Please enter a valid file path or name.")

    def get_target_column(self):
        """
        Prompt the user to enter the name of the dependent target column in the dataset.

        Returns:
            None
        """
        # Display columns in dataset and get target column from the user.
        print("Columns in the data:", self.df.columns)
        self.target_col = input("Enter the dependent target column: ")
        if self.target_col not in self.df.columns:
            print("Invalid target column.")
            self.target_col = input("Choose one of the columns in the data: ")

    def preprocess_data(self):
        """
        Preprocess the data by handling missing values and converting categorical features.

        Returns:
            None
        """
        # Check if there are missing values in the dataset.
        if self.df.isnull().sum().sum() > 0:
            raise ValueError \
                ("There are missing values in the data. \
                 Please, fill them and rerun the app.")

        # Identify categorical columns in the dataset.
        cat_cols = self.df.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            print("Data is not ready for machine learning process.")
            convert = input("There are categorical/string values in the data. "
                            "Do you like to convert them? Enter yes or no: ")
            if convert.lower() == 'yes':
                # Convert categorical features to binary using one-hot encoding.
                self.df = pd.get_dummies(self.df, columns=cat_cols, drop_first=True)
                print("Categorical/string values are now converted.")
            elif convert.lower() == 'no':
                raise ValueError("Data is not ready for machine learning process. "
                                "There are categorical/string values in the data. "
                                "Convert them and rerun the app.")
            else:
                raise ValueError("Invalid response. Please enter 'yes' or 'no'.")

    def run_machine_learning(self):
        """
        Run machine learning tasks based on the selected ML type (regressor or classifier).

        Returns:
            None
        """
        if self.ml_type == 'regressor':
            # Create an instance of the RegressionCases class
            self.ml_regressor = RegressionCases(self.df, self.target_col)
            # Perform the machine learning tasks using the RegressionCases class
            self.ml_regressor.train_models()
            self.ml_regressor.grid_search()
            self.ml_regressor.evaluate_models()
        elif self.ml_type == 'classifier':
            # Create an instance of the ClassificationCases class
            self.ml_classifier = ClassificationCases(self.df, self.target_col)
            # Perform the machine learning tasks using the ClassificationCases class
            self.ml_classifier.split_data()
            self.ml_classifier.scale_data()
            self.ml_classifier.train_models()
            self.ml_classifier.grid_search()
            self.ml_classifier.evaluate_models()



if __name__ == '__main__':
    # Create an instance of the ML_Wizard class
    my_model = ML_Wizard()
    # Prompt user to select ML type (regressor or classifier)
    my_model.get_ml_type()
    # Prompt user to enter dataset file
    my_model.read_csv_file()
    # Prompt user to enter the target column
    my_model.get_target_column()
    # Preprocess the dataset
    my_model.preprocess_data()
    # Run machine learning tasks based on user input
    my_model.run_machine_learning()