from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import pickle

# Load built-in dataset
cancer = load_breast_cancer()

# Create dataframe
df = pd.DataFrame(
    cancer.data,
    columns=cancer.feature_names
)

# Rename columns
df.columns = [col.replace(' ', '_') for col in df.columns]

# Select important features
important_features = [
    'mean_radius',
    'mean_texture',
    'mean_perimeter',
    'mean_area',
    'mean_smoothness',
    'mean_concavity'
]

# Input and target
X = df[important_features]
y = cancer.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier()

model.fit(X_train, y_train)

# Save files
pickle.dump(model, open('breast_cancer_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
pickle.dump(important_features, open('feature_names.pkl', 'wb'))

print("Model Trained Successfully")