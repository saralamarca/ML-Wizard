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
    def __init__(self, df, target_col) -> None:
        self.df = df
        self.target_col = target_col

        # Create X and y
        X = df.drop(target_col, axis=1)
        y = df[target_col]

        # If X is only 1 column - reshape
        if len(X.columns) == 1:
            column_name = X.columns[0]
            X = X[column_name].values.reshape(-1, 1)

def reg_ml(df, ml_type, target_col):
    # If the dataset is ready for machine learning process and 
    # ml_type == 'regressor' - Create X and y
    if ml_type == 'regressor':
        X = df.drop(target_col, axis=1)
        y = df[target_col]

        # If X is only 1 column - reshape
        if len(X.columns) == 1:
            column_name = X.columns[0]
            X = X[column_name].values.reshape(-1, 1)

        # Split data into train and test sets based on how many rows in the dataset
        if len(X) < 1000:
            test_size = 0.3
        elif len(X) < 2000:
            test_size = 0.15
        else:
            test_size = 0.1
        X_train, X_test, y_train, y_test = train_test_split(X, y, \
                                test_size=test_size, random_state=101)

        # Scale data
        scaler = StandardScaler()
        scaled_X_train = scaler.fit_transform(X_train)
        scaled_X_test = scaler.transform(X_test)

        # Create a Linear Regression model
        lir_model = LinearRegression()
        lir_model.fit(scaled_X_train,y_train)
        lir_pred = lir_model.predict(scaled_X_test)

        # Create a Lasso model
        lasso_model = Lasso()
        lasso_model.fit(scaled_X_train,y_train)
        lasso_pred = lasso_model.predict(scaled_X_test)

        # Create a Ridge model
        ridge_model = Ridge()
        ridge_model.fit(scaled_X_train,y_train)
        ridge_pred = ridge_model.predict(scaled_X_test)

        # Create a Elastic Net model
        elastic_model = ElasticNet()
        elastic_model.fit(scaled_X_train,y_train)
        elastic_pred = elastic_model.predict(scaled_X_test)

        # Create a SVR model
        svr_model = SVR()
        svr_model.fit(scaled_X_train,y_train)
        svr_pred = svr_model.predict(scaled_X_test)

        # Parameter grid to use for GridSearchCV for each model
        lir_param_grid = {'fit_intercept': [True, False],'copy_X': \
                [True, False],'positive': [True, False]}
        lasso_param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0]}
        ridge_param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0], 'solver': \
                ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}
        elastic_param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0], 'l1_ratio': \
                [0.1, 0.5, 0.9], 'max_iter': [1000, 5000, 10000]}
        svr_param_grid = {'C': [0.1, 1, 10], 'kernel': \
                ['linear', 'rbf', 'sigmoid']}

        # GridSearchCV for each model
        lir_grid = GridSearchCV(estimator=lir_model, \
                                param_grid=lir_param_grid)
        lasso_grid = GridSearchCV(estimator=lasso_model, \
                                  param_grid=lasso_param_grid)
        ridge_grid = GridSearchCV(estimator=ridge_model, \
                                  param_grid=ridge_param_grid)
        elastic_grid = GridSearchCV(estimator=elastic_model, \
                                    param_grid=elastic_param_grid)
        svr_grid = GridSearchCV(estimator=svr_model, \
                                param_grid=svr_param_grid)
        
        # Fit and get best parameters for each model
        lir_grid.fit(scaled_X_train,y_train)
        best_lir = [lir_grid.best_params_]
        lasso_grid.fit(scaled_X_train,y_train)
        best_lasso = [lasso_grid.best_params_]
        ridge_grid.fit(scaled_X_train,y_train)
        best_ridge = [ridge_grid.best_params_]
        elastic_grid.fit(scaled_X_train,y_train)
        best_elastic = [elastic_grid.best_params_]
        svr_grid.fit(scaled_X_train,y_train)
        best_svr = [svr_grid.best_params_]

        # Calculate MAE
        lir_MAE = mean_absolute_error(y_test,lir_pred)
        lasso_MAE = mean_absolute_error(y_test,lasso_pred)
        ridge_MAE = mean_absolute_error(y_test,ridge_pred)
        elastic_MAE = mean_absolute_error(y_test,elastic_pred)
        svr_MAE = mean_absolute_error(y_test,svr_pred)

        # Calculate RMSE
        lir_RMSE = np.sqrt(mean_squared_error(y_test,lir_pred))
        lasso_RMSE = np.sqrt(mean_squared_error(y_test,lasso_pred))
        ridge_RMSE = np.sqrt(mean_squared_error(y_test,ridge_pred))
        elastic_RMSE = np.sqrt(mean_squared_error(y_test,elastic_pred))
        svr_RMSE = np.sqrt(mean_squared_error(y_test,svr_pred))

        # Calculate R2 Score
        lir_score = r2_score(y_test, lir_pred)
        lasso_score = r2_score(y_test, lasso_pred)
        ridge_score = r2_score(y_test, ridge_pred)
        elastic_score = r2_score(y_test, elastic_pred)
        svr_score = r2_score(y_test, svr_pred)
        
        # Print best hyperparameters for each model
        print("\n\nBest hyperparameters for Linear Regression: ", best_lir), \
        print("Best hyperparameters for Lasso: ", best_lasso), \
        print("Best hyperparameters for Ridge: ", best_ridge), \
        print("Best hyperparameters for Elastic: ", best_elastic), \
        print("Best hyperparameters for SVR: ", best_svr), \
        print("\nMAE for Linear Regression model: ", lir_MAE), \
        print("MAE for Lasso model: ", lasso_MAE), \
        print("MAE for Ridge model: ", ridge_MAE), \
        print("MAE for ElasticNet model: ", elastic_MAE), \
        print("MAE for SVR model: ", svr_MAE), \
        print("\nRMSE for Linear Regression model: ", lir_RMSE), \
        print("RMSE for Lasso model: ", lasso_RMSE), \
        print("RMSE for Ridge model: ", ridge_RMSE), \
        print("RMSE for ElasticNet model: ", elastic_RMSE), \
        print("RMSE for SVR model: ", svr_RMSE), \
        print("\nScore for Linear Regression model: ", lir_score), \
        print("Score for Lasso model: ", lasso_score), \
        print("Score for Ridge model: ", ridge_score), \
        print("Score for ElasticNet model: ", elastic_score), \
        print("Score for SVR model: ", svr_score)

        # Print some feedback/conclusion
        scores = [lir_score, lasso_score, ridge_score, \
                                elastic_score, svr_score]
        highest_score = max(scores)
        if highest_score == lir_score:
            best_model = "Linear Regression"
            print(f"\n\nBest model with highest score for your data is: {best_model}")
        elif highest_score == lasso_score:
            best_model = "Lasso model"
            print(f"\n\nBest model with highest score for your data is: {best_model}")
        elif highest_score == ridge_score:
            best_model = "Ridge model"
            print(f"\n\nBest model with highest score for your data is: {best_model}")
        elif highest_score == elastic_score:
            best_model = "ElasticNet model"
            print(f"\n\nBest model with highest score for your data is: {best_model}")
        elif highest_score == svr_score:
            best_model = "SVR model"
            print(f"\n\nBest model with highest score for your data is: {best_model}")
                
        # Save model, yes or no
        save_model = input("\nDo you want to save model? Enter yes or no: ")
        if save_model.lower() == 'yes':
            model_name = input \
            ("\nEnter what you want to name your model: (format example xxx.joblib) ")
            dump(best_model, model_name)
            print("Model saved.")
        elif save_model.lower() == 'no':
            print("Model not saved.")
            exit()
        else:
            print("Invalid response. Please enter 'yes' or 'no'")