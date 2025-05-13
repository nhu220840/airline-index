import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from ydata_profiling import ProfileReport
import joblib

data = pd.read_csv('data/raw_data.csv')
# profile = ProfileReport(data, title="Invistico Airline Report")
# profile.to_file('airline-report.html')

rename_mapping = {
    "satisfaction": "satisfaction",
    "Gender": "gender",
    "Customer Type": "cust_type",
    "Age": "age",
    "Type of Travel": "travel_type",
    "Class": "class",
    "Flight Distance": "flight_dist",
    "Seat comfort": "seat",
    "Departure/Arrival time convenient": "time_conv",
    "Food and drink": "food",
    "Gate location": "gate_loc",
    "Inflight wifi service": "wifi",
    "Inflight entertainment": "entertain",
    "Online support": "support",
    "Ease of Online booking": "booking",
    "On-board service": "onboard",
    "Leg room service": "legroom",
    "Baggage handling": "baggage",
    "Checkin service": "checkin",
    "Cleanliness": "clean",
    "Online boarding": "online_board",
    "Departure Delay in Minutes": "dep_delay",
    "Arrival Delay in Minutes": "arr_delay"
}

data = data.rename(columns=rename_mapping)

target = "satisfaction"
x = data.drop(target, axis=1)
y = data[target]

# train: 70 - validattion: 15 - test: 15
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.30, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.50, random_state=42)

# for col in data.columns:
#     print(f"{col}: {data[col].unique()}")

# print(data.isnull().sum())

num_features = ['age', 'flight_dist', 'dep_delay', 'arr_delay','seat',
                'time_conv', 'food', 'gate_loc', 'wifi', 'entertain', 'support',
                'booking', 'onboard', 'legroom', 'baggage', 'checkin', 'clean', 'online_board']
nom_features = ['gender', 'cust_type', 'travel_type', 'class']

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

nom_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num_feature", num_transformer, num_features),
    ("nom_feature", nom_transformer, nom_features),
])

x_train_processed = preprocessor.fit_transform(x_train)
x_val_processed = preprocessor.transform(x_val)
x_test_processed = preprocessor.transform(x_test)

nom_cols = preprocessor.named_transformers_['nom_feature'].named_steps['encoder'].get_feature_names_out(nom_features)
all_cols = np.concatenate([num_features, nom_cols])

data_train = pd.DataFrame(x_train_processed, columns=all_cols)
data_val = pd.DataFrame(x_val_processed, columns=all_cols)
data_test = pd.DataFrame(x_test_processed, columns=all_cols)

data_train[target] = y_train.reset_index(drop=True)
data_val[target] = y_val.reset_index(drop=True)
data_test[target] = y_test.reset_index(drop=True)

data_train.to_csv("data/supervised/train_data.csv", index=False)
data_val.to_csv("data/supervised/val_data.csv", index=False)
data_test.to_csv("data/supervised/test_data.csv", index=False)

joblib.dump(preprocessor, "data/supervised/preprocessor.pkl")