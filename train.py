import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from joblib import dump

df = pd.read_csv('train.csv')

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('model', LinearRegression())
])

X = df.drop('target', axis=1)
y = df['target']

pipeline.fit(X, y)
dump(pipeline, 'model.pkl')
