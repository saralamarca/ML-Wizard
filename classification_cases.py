import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error, mean_squared_error, \
    confusion_matrix, classification_report, accuracy_score
from joblib import dump



class ClassificationCases:
    """
    A class for performing classification tasks on a dataset.

    This class provides methods for data preprocessing, model training, hyperparameter tuning,
    model evaluation and model saving.

    Args:
        df (pd.DataFrame): The input DataFrame containing the dataset.
        target_col (str): The name of the target column to predict.
    """
    def __init__(self, df, target_col):
        """
        Initialize the ClassificationCases instance.

        Args:
            df (pd.DataFrame): The input DataFrame containing the dataset.
            target_col (str): The name of the target column to predict.
        """
        # Initialize instance variables
        self.df = df
        self.target_col = target_col

        # Create X and y
        self.X = self.df.drop(target_col, axis=1)
        self.y = self.df[target_col]

        # If X is only 1 column - Reshape
        if len(self.X.columns) == 1:
            column_name = self.X.columns[0]
            self.X = self.X[column_name].values.reshape(-1, 1)
    
    def split_data(self):
        """
        Split the dataset into training and testing sets.

        The size of the test set is determined based on the number of rows in the dataset.

        Returns:
            tuple: A tuple containing X_train, X_test, y_train, and y_test.
        """
        # Determine the size of the test set based on the dataset's row count.
        if len(self.X) < 1000:
            test_size = 0.3  # If the dataset has less than 1000 rows, set the test size to 30%.
        elif len(self.X) < 2000:
            test_size = 0.15  # If the dataset has 1000-1999 rows, set the test size to 15%.
        else:
            test_size = 0.1  # If the dataset has 2000 or more rows, set the test size to 10%.

        # Use train_test_split to split the data into training and testing sets.
        # random_state=101 ensures reproducibility of the split.
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=101)

        # Return the split datasets as a tuple.
        return self.X_train, self.X_test, self.y_train, self.y_test

    def scale_data(self):
        """
        Standardize the features by scaling them.

        Returns:
            tuple: A tuple containing scaled_X_train and scaled_X_test.
        """
        # Create a StandardScaler instance
        scaler = StandardScaler()
        # Scale the training data using the scaler
        self.scaled_X_train = scaler.fit_transform(self.X_train)
        # Transform the test data using the same scaler
        self.scaled_X_test = scaler.transform(self.X_test)
        # Return a tuple containing scaled training and test data
        return self.scaled_X_train, self.scaled_X_test

    def train_models(self):
        """
        Initializes and trains the following classifier models:
        - Logistic Regression
        - K-Nearest Neighbors (KNN)
        - Support Vector Classifier (SVC)

        The trained models are stored as instance variables for later use.

        Returns:
            None
        
        """
        # Initialize each models
        self.log_model = LogisticRegression()
        self.knn_model = KNeighborsClassifier()
        self.svc_model = SVC()

        # Train each model using the scaled training data
        self.log_model.fit(self.scaled_X_train, self.y_train)
        self.knn_model.fit(self.scaled_X_train, self.y_train)
        self.svc_model.fit(self.scaled_X_train, self.y_train)

    def grid_search(self):
        """
        Perform grid search for hyperparameter tuning of the classifier models.

        This method performs grid search to find the best hyperparameters for the
        Logistic Regression, K-Nearest Neighbors (KNN) and Support Vector Classifier (SVC)
        models using the specified parameter grids.

        The best hyperparameters for each model are stored in the attributes self.best_log,
        self.best_knn and self.best_svc.

        Returns:
            None
        """
        # Define parameter grids for grid search
        log_param_grid = {'penalty': ['l1', 'l2', 'elasticnet', 'None'],
                          'C': [0.1, 1, 10, 100],
                          'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                          'max_iter': [100, 1000, 5000]}

        knn_param_grid = {'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
                          'weights': ['uniform', 'distance'],
                          'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

        svc_param_grid = {'C': [0.1, 1, 10, 100],
                          'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                          'gamma': ['scale', 'auto', 0.1, 1, 10]}

        # Create GridSearchCV objects for each model
        self.log_grid = GridSearchCV(estimator=LogisticRegression(), param_grid=log_param_grid)
        self.knn_grid = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=knn_param_grid)
        self.svc_grid = GridSearchCV(estimator=SVC(), param_grid=svc_param_grid)

        # Fit GridSearchCV for each model using scaled training data
        self.log_grid.fit(self.scaled_X_train, self.y_train)
        self.knn_grid.fit(self.scaled_X_train, self.y_train)
        self.svc_grid.fit(self.scaled_X_train, self.y_train)

        # Get best hyperparameters for each model
        self.best_log = self.log_grid.best_params_
        self.best_knn = self.knn_grid.best_params_
        self.best_svc = self.svc_grid.best_params_
        print(f"Best hyperparameters for Logistic Regression: {self.best_log}\n")
        print(f"Best hyperparameters for K-Nearest Neighbors: {self.best_knn}\n")
        print(f"Best hyperparameters for Support Vector Classification : {self.best_svc}\n")

    def evaluate_models(self):
        """
        Evaluates the Logistic Regression, K-Nearest Neighbors (KNN),
        and Support Vector Classifier (SVC) models on the test data and displays
        confusion matrices, classification reports and accuracy scores.

        It then identifies the best-performing model and offers the option to save it.
        If the user chooses to save the model, they will be prompted to enter a filename.

        Returns:
            None
        """
        # Predictions for each model
        log_pred = self.log_model.predict(self.scaled_X_test)
        knn_pred = self.knn_model.predict(self.scaled_X_test)
        svc_pred = self.svc_model.predict(self.scaled_X_test)

        # Calculate confusion matrices for each model
        log_con_matrix = confusion_matrix(self.y_test, log_pred)
        knn_con_matrix = confusion_matrix(self.y_test, knn_pred)
        svc_con_matrix = confusion_matrix(self.y_test, svc_pred)

        # Print confusion matrices for each model
        print(f"\nConfusion matrix for Logistic Regression model:\n{log_con_matrix}\n")
        print(f"Confusion matrix for KNN Model:\n{knn_con_matrix}\n")
        print(f"Confusion matrix for SVC Model:\n{svc_con_matrix}\n")

        # Print classification report for each model
        print("Classification report for Logistic Regression model:\n", classification_report(self.y_test, log_pred))
        print("\nClassification report for KNN model:\n", classification_report(self.y_test, knn_pred))
        print("\nClassification report for SVC model:\n", classification_report(self.y_test, svc_pred))

        # Calculate accuracy score for each model
        log_acc = accuracy_score(self.y_test, log_pred)
        knn_acc = accuracy_score(self.y_test, knn_pred)
        svc_acc = accuracy_score(self.y_test, svc_pred)

        # Print accuracy score for each model
        print("Logistic Regression model accuracy:", log_acc)
        print("KNN model accuracy:", knn_acc)
        print("SVC model accuracy:", svc_acc)

        # Get and print the model with highest score
        scores = [log_acc, knn_acc, svc_acc]
        highest_score = max(scores)
        self.best_model = best_model
        if highest_score == log_acc:
            best_model = "Logistic Regression"
        elif highest_score == knn_acc:
            best_model = "KNN model"
        elif highest_score == svc_acc:
            best_model = "SVC model"
        print(f"\nBest model with highest score for your data is: {self.best_model}")

        # Option to save the best model
        save_model = input("\nDo you want to save model? Enter yes or no: ")
        if save_model.lower() == 'yes':
            model_name = input("\nEnter what you want to name your model (format example xxx.joblib): ")
            dump(self.best_model, model_name)
            print("Model saved.")
        elif save_model.lower() == 'no':
            print("Model not saved.")
        else:
            print("Invalid response. Please enter 'yes' or 'no'")