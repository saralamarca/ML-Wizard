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
from regression_cases import reg_ml
from classification_cases import class_ml



# Read in from user what type of supervised ML
# they need - regressor or classifier
while True:
    ml_type = input\
        ("What kind of supervised machine learning model do you need? \
        \nEnter 'regressor' or 'classifier': ")
    if ml_type not in ['regressor', 'classifier']:
        print("Invalid input.")
    else:
        break

# Read in .csv file
while True:
    file_name = input \
        ("Enter the path or .csv file name: (format example xxx.csv) ")
    try:
        df = pd.read_csv(file_name)
        break
    except FileNotFoundError:
        print(f"File not found at {file_name}. \
            \nPlease enter a valid file path or name.")

# Print all columns to user
# Read in dependent target (y) and validate if choice is right
print("Columns in the data:", df.columns)
target_col = input("Enter the dependent target column: ")
if df[target_col].dtype == [np.int64, np.float64]:
    print("\nDependent target is continuous.")
elif df[target_col].dtype == 'object':
    print("\nDependent target is categorical.")
elif target_col not in df.columns:
    print("Invalid target column.")
    target_col = input("Choose one of the columns in the data: ", df.columns)


# Check if data is ready for machine learning process or not. Any NaN values?
if df.isnull().sum().sum() > 0:
    print("\nData is not ready for machine learning process.")
    print("There are missing values in the data. \
        \nPlease, fill them and rerun the app.")
    exit()


# Check if data is ready for the machine learning process or not. \
# Any categroical values?
cat_cols = df.select_dtypes(include=['object']).columns
if len(cat_cols) > 0:
    print("Data is not ready for machine learning process.")
    convert = input("There are categorical/string values in the data. \
                    \nDo you like to convert them? Enter yes or no: ")
    if convert.lower() == 'yes':
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        print("Categorical/string values are now converted.")
    elif convert.lower() == 'no':
        print("Data is not ready for machine learning process.")
        print("There are categorical/string values in the data. \
            \nConvert them and rerun the app.")
        exit()
    else:
        print("Invalid response. Please enter 'yes' or 'no'.")
        exit()
else:
    print("Data is ready for machine learning process.")

reg_ml(df, ml_type, target_col)
class_ml(df, ml_type, target_col)