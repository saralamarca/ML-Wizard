import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump



class RegressionCases:
    """
    A class for performing regression tasks on a dataset.

    This class provides methods for data preprocessing, model training, hyperparameter tuning,
    model evaluation and model saving.

    Args:
        df (pd.DataFrame): The input DataFrame containing the dataset.
        target_col (str): The name of the target column to predict.
    """
    def __init__(self, df, target_col):
        """
        Initialize the RegressionCases instance.

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
        
        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()

        # Scale the feature data
        self.scaled_X_train, self.scaled_X_test = self.scale_data()

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
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=101)

        # Return the split datasets as a tuple.
        return X_train, X_test, y_train, y_test

    def scale_data(self):
        """
        Standardize the features by scaling them.

        Returns:
            tuple: A tuple containing scaled_X_train and scaled_X_test.
        """
        # Create a StandardScaler instance
        scaler = StandardScaler()
        # Scale the training data using the scaler
        scaled_X_train = scaler.fit_transform(self.X_train)
        # Transform the test data using the same scaler
        scaled_X_test = scaler.transform(self.X_test)
        # Return a tuple containing scaled training and test data
        return scaled_X_train, scaled_X_test

    def train_models(self):
        """
        Initializes and trains the following regression models:
        - Linear Regression
        - Lasso Regression
        - Ridge Regression
        - ElasticNet Regression
        - Support Vector Regression (SVR)

        The trained models are stored as instance variables for later use.

        Returns:
            None
        """
        # Initialize regression models
        self.lir_model = LinearRegression()
        self.lasso_model = Lasso()
        self.ridge_model = Ridge()
        self.elastic_model = ElasticNet()
        self.svr_model = SVR()

        # Train each model using the scaled training data
        self.lir_model.fit(self.scaled_X_train, self.y_train)
        self.lasso_model.fit(self.scaled_X_train, self.y_train)
        self.ridge_model.fit(self.scaled_X_train, self.y_train)
        self.elastic_model.fit(self.scaled_X_train, self.y_train)
        self.svr_model.fit(self.scaled_X_train, self.y_train)

    def grid_search(self):
        """
        Performs hyperparameter tuning using GridSearchCV for the Linear Regression, Lasso, Ridge, ElasticNet,
        and SVR models.

        Returns:
            None
        """
        # Define parameter grids for grid search
        lir_param_grid = {'fit_intercept': [True, False], 'copy_X': [True, False], 'positive': [True, False]}
        lasso_param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0]}
        ridge_param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0], 'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}
        elastic_param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0], 'l1_ratio': [0.1, 0.5, 0.9], 'max_iter': [1000, 5000, 10000]}
        svr_param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'sigmoid']}

        # Create GridSearchCV objects for each model
        self.lir_grid = GridSearchCV(estimator=self.lir_model, param_grid=lir_param_grid)
        self.lasso_grid = GridSearchCV(estimator=self.lasso_model, param_grid=lasso_param_grid)
        self.ridge_grid = GridSearchCV(estimator=self.ridge_model, param_grid=ridge_param_grid)
        self.elastic_grid = GridSearchCV(estimator=self.elastic_model, param_grid=elastic_param_grid)
        self.svr_grid = GridSearchCV(estimator=self.svr_model, param_grid=svr_param_grid)

        # Fit GridSearchCV for each model
        self.lir_grid.fit(self.scaled_X_train, self.y_train)
        self.lasso_grid.fit(self.scaled_X_train, self.y_train)
        self.ridge_grid.fit(self.scaled_X_train, self.y_train)
        self.elastic_grid.fit(self.scaled_X_train, self.y_train)
        self.svr_grid.fit(self.scaled_X_train, self.y_train)

        # Get best hyperparameters for each model
        self.best_lir = self.lir_grid.best_params_
        self.best_lasso = self.lasso_grid.best_params_
        self.best_ridge = self.ridge_grid.best_params_
        self.best_elastic = self.elastic_grid.best_params_
        self.best_svr = self.svr_grid.best_params_

    def evaluate_models(self):
        """
        Calculates Mean Absolute Error (MAE), Root Mean Squared Error (RMSE) and R-squared (R2) scores for each
        trained regression model. It then identifies the best-performing model and offers the option to save it.

        If the user chooses to save the model, they will be prompted to enter a filename.

        Returns:
            None
        """
        # Predictions for each model
        lir_pred = self.lir_model.predict(self.scaled_X_test)
        lasso_pred = self.lasso_model.predict(self.scaled_X_test)
        ridge_pred = self.ridge_model.predict(self.scaled_X_test)
        elastic_pred = self.elastic_model.predict(self.scaled_X_test)
        svr_pred = self.svr_model.predict(self.scaled_X_test)

        # Calculate MAE for each model
        lir_MAE = mean_absolute_error(self.y_test, lir_pred)
        lasso_MAE = mean_absolute_error(self.y_test, lasso_pred)
        ridge_MAE = mean_absolute_error(self.y_test, ridge_pred)
        elastic_MAE = mean_absolute_error(self.y_test, elastic_pred)
        svr_MAE = mean_absolute_error(self.y_test, svr_pred)

        # Calculate RMSE for each model
        lir_RMSE = np.sqrt(mean_squared_error(self.y_test, lir_pred))
        lasso_RMSE = np.sqrt(mean_squared_error(self.y_test, lasso_pred))
        ridge_RMSE = np.sqrt(mean_squared_error(self.y_test, ridge_pred))
        elastic_RMSE = np.sqrt(mean_squared_error(self.y_test, elastic_pred))
        svr_RMSE = np.sqrt(mean_squared_error(self.y_test, svr_pred))

        # Calculate R2 Score for each model
        lir_score = r2_score(self.y_test, lir_pred)
        lasso_score = r2_score(self.y_test, lasso_pred)
        ridge_score = r2_score(self.y_test, ridge_pred)
        elastic_score = r2_score(self.y_test, elastic_pred)
        svr_score = r2_score(self.y_test, svr_pred)

        # Print results
        print("\nMAE for Linear Regression model:", lir_MAE)
        print("MAE for Lasso model:", lasso_MAE)
        print("MAE for Ridge model:", ridge_MAE)
        print("MAE for ElasticNet model:", elastic_MAE)
        print("MAE for SVR model:", svr_MAE)

        print("\nRMSE for Linear Regression model:", lir_RMSE)
        print("RMSE for Lasso model:", lasso_RMSE)
        print("RMSE for Ridge model:", ridge_RMSE)
        print("RMSE for ElasticNet model:", elastic_RMSE)
        print("RMSE for SVR model:", svr_RMSE)

        print("\nScore for Linear Regression model:", lir_score)
        print("Score for Lasso model:", lasso_score)
        print("Score for Ridge model:", ridge_score)
        print("Score for ElasticNet model:", elastic_score)
        print("Score for SVR model:", svr_score)

        # Identify and print the best model
        scores = [lir_score, lasso_score, ridge_score, elastic_score, svr_score]
        best_model_idx = scores.index(max(scores))
        best_models = ["Linear Regression", "Lasso model", "Ridge model", "ElasticNet model", "SVR model"]
        best_model = best_models[best_model_idx]
        print(f"\nBest model with the highest score for your data is: {best_model}")

        # Option to save the best model
        save_model = input("\nDo you want to save the best model? Enter yes or no: ")
        if save_model.lower() == 'yes':
            model_name = input("\nEnter what you want to name your model (format example xxx.joblib): ")
            if model_name:
                if best_model == "Linear Regression":
                    dump(self.lir_model, model_name)
                elif best_model == "Lasso model":
                    dump(self.lasso_model, model_name)
                elif best_model == "Ridge model":
                    dump(self.ridge_model, model_name)
                elif best_model == "ElasticNet model":
                    dump(self.elastic_model, model_name)
                elif best_model == "SVR model":
                    dump(self.svr_model, model_name)
                print(f"Model ({best_model}) saved as {model_name}.")
            else:
                print("Invalid filename.")
        elif save_model.lower() == 'no':
            print("Model not saved.")
        else:
            print("Invalid response. Please enter 'yes' or 'no'")