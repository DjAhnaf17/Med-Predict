import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('disease.csv')

# Features and target
SYMPTOMS = ['Fever', 'Cough', 'Headache', 'Fatigue', 'Nausea',
            'Vomiting', 'Rash', 'Joint_Pain', 'Sore_Throat', 'Shortness_of_Breath']

X = df[SYMPTOMS]
y = df['Disease']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Train models
models = {
    'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'svm': SVC(kernel='rbf', probability=True, random_state=42),
    'naive_bayes': GaussianNB()
}

os.makedirs('models', exist_ok=True)

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name}: {acc*100:.1f}% accuracy")
    with open(f'models/{name}.pkl', 'wb') as f:
        pickle.dump(model, f)

# Save label encoder and symptoms list
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

with open('models/symptoms.pkl', 'wb') as f:
    pickle.dump(SYMPTOMS, f)

print("All models saved successfully!")
print("Diseases:", le.classes_.tolist())
