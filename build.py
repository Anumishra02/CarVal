import pandas as pd
import numpy as np
import pickle
import json
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

print("Training model...")
df = pd.read_csv('dataset.csv')

df['Mileage'] = df['Mileage'].str.extract(r'([\d.]+)').astype(float)
df['Engine']  = df['Engine'].str.extract(r'([\d.]+)').astype(float)
df['Power']   = df['Power'].str.extract(r'([\d.]+)').astype(float)
df.drop(columns=['Unnamed: 0', 'New_Price'], inplace=True, errors='ignore')
df.dropna(inplace=True)
df['car_age']      = 2024 - df['Year']
df['kms_per_year'] = df['Kilometers_Driven'] / (df['car_age'] + 1)

X = df[['Name','Location','Year','car_age','Kilometers_Driven',
        'kms_per_year','Fuel_Type','Transmission','Owner_Type',
        'Mileage','Engine','Power','Seats']]
y = df['Price']

cat_cols = ['Name','Location','Fuel_Type','Transmission','Owner_Type']

ct = ColumnTransformer([
    ('ohe', OneHotEncoder(handle_unknown='ignore'), cat_cols)
], remainder='passthrough')

model = make_pipeline(ct, GradientBoostingRegressor(
    n_estimators=500, max_depth=5,
    learning_rate=0.05, subsample=0.8, random_state=42
))

model.fit(X, y)
pickle.dump(model, open('GBModel_v2.pkl', 'wb'))
print("Model saved!")

# Save meta
df['company'] = df['Name'].str.split().str[0]
company_cars = {}
for c in sorted(df['company'].unique()):
    company_cars[c] = sorted(df[df['company'] == c]['Name'].unique().tolist())

meta = {
    'locations': sorted(df['Location'].unique().tolist()),
    'fuel_types': sorted(df['Fuel_Type'].unique().tolist()),
    'transmissions': sorted(df['Transmission'].unique().tolist()),
    'owner_types': df['Owner_Type'].unique().tolist(),
    'year_min': int(df['Year'].min()),
    'year_max': int(df['Year'].max()),
    'seats_options': sorted(df['Seats'].dropna().unique().astype(int).tolist()),
    'companies': sorted(df['company'].unique().tolist()),
    'company_cars': company_cars
}
with open('meta_v2.json', 'w') as f:
    json.dump(meta, f)
print("Meta saved!")
print("Build complete!")