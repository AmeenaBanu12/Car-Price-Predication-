import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import joblib

data = pd.read_csv('car_price_prediction.csv')

X = data.drop('Price', axis=1)
y = data['Price']

cat_cols = X.select_dtypes(include='object').columns
num_cols = X.select_dtypes(exclude='object').columns

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ('num', 'passthrough', num_cols)
])

model = Pipeline([
    ('prep', preprocessor),
    ('lr', LinearRegression())
])

model.fit(X, y)

joblib.dump(model, 'car_price_model.pkl')

print("Model trained and saved")
