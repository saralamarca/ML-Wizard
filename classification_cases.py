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


def class_ml(df, ml_type, target_col):
    # If data is ready for the machine learning process and ML type is classifier
    if ml_type == 'classifier':
        # Create X and y
        X = df.drop(target_col, axis=1)
        y = df[target_col]

        # If X is only 1 column - reshape
        if len(X.columns) == 1:
            column_name = X.columns[0]
            X = X[column_name].values.reshape(-1, 1)

        # Split data into train/test based on how many rows in dataset
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

        # Logistic Regrssion
        log_model = LogisticRegression()
        log_model.fit(scaled_X_train,y_train)
        log_pred = log_model.predict(scaled_X_test)

        # KNeighborsClassifier
        knn_model = KNeighborsClassifier()
        knn_model.fit(scaled_X_train,y_train)
        knn_pred = knn_model.predict(scaled_X_test)

        # SVC
        SVC_model = SVC ()
        SVC_model.fit(scaled_X_train,y_train)
        SVC_pred = SVC_model.predict(scaled_X_test)

        # param_grid for each model
        log_param_grid = {'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                'C': [0.1, 1, 10, 100],
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                'max_iter': [100, 1000, 5000]}

        knn_param_grid = {'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

        svc_param_grid = {'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'gamma': ['scale', 'auto', 0.1, 1, 10]}

        # GridSearchCV for each model
        log_grid = GridSearchCV(estimator=LogisticRegression(), \
                                param_grid=log_param_grid)
        knn_grid = GridSearchCV(estimator=KNeighborsClassifier(), \
                                param_grid=knn_param_grid)
        svc_grid = GridSearchCV(estimator=SVC(), \
                                param_grid=svc_param_grid)

        # fit and get best_params for each model
        log_grid.fit(scaled_X_train, y_train)
        best_log = [log_grid.best_params_]
        knn_grid.fit(scaled_X_train, y_train)
        best_knn = [knn_grid.best_params_]
        svc_grid.fit(scaled_X_train, y_train)
        best_svc = [svc_grid.best_params_]

        # Plot confusion matrix for each model
        log_con_matrix = confusion_matrix(y_test,log_pred)
        knn_con_matrix = confusion_matrix(y_test,knn_pred)
        SVC_con_matrix = confusion_matrix(y_test,SVC_pred)
        print(f"\nConfusion matrix for Logistic Regression model: \
                                            \n{log_con_matrix}\n"
        f"\nConfusion matrix for KNN Model: \n{knn_con_matrix}\n"
        f"\nConfusion matrix for SVC Model: \n{SVC_con_matrix}\n")

        # Print classification report per each.
        print("Classification report for Logistic Regression model: \n", \
              classification_report(y_test,log_pred))
        print("\n\nClassification report for KNN model: \n", \
              classification_report(y_test,knn_pred))
        print("\n\nClassification report for SVC model: \n", \
              classification_report(y_test,SVC_pred))

        # Accuracy for each model
        log_acc = accuracy_score(y_test, log_pred)
        print("Logistic Regression model accuracy: ", log_acc)
        knn_acc = accuracy_score(y_test, knn_pred)
        print("KNN model accuracy: ", knn_acc)
        svc_acc = accuracy_score(y_test, SVC_pred)
        print("SVC model accuracy: ", svc_acc)

        # Feedback/conclusion
        scores = [log_acc, knn_acc, svc_acc]
        highest_score = max(scores)
        if highest_score == log_acc:
            best_model = "Logistic Regression"
            print(f"\n\nBest model with highest score for your data is: {best_model}")
        elif highest_score == knn_acc:
            best_model = "KNN model"
            print(f"\n\nBest model with highest score for your data is: {best_model}")
        elif highest_score == svc_acc:
            best_model = "SVC model"
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