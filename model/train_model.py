import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Create a synthetic clinical dataset for testing
def generate_mock_data():
    np.random.seed(42)
    n_samples = 1000
    
    # Simulating clinical features
    age = np.random.randint(18, 80, size=n_samples)
    tsh = np.random.uniform(0.1, 15.0, size=n_samples)
    t3 = np.random.uniform(0.5, 4.5, size=n_samples)
    t4 = np.random.uniform(4.0, 22.0, size=n_samples)
    
    # Simplified clinical logic for classification
    # 0: Normal (Euthyroid), 1: Hyperthyroidism, 2: Hypothyroidism
    target = np.zeros(n_samples)
    for i in range(n_samples):
        if tsh[i] < 0.4 and t4[i] > 12.0:
            target[i] = 1 # Hyperthyroidism
        elif tsh[i] > 4.5 and t4[i] < 5.0:
            target[i] = 2 # Hypothyroidism
            
    df = pd.DataFrame({
        'Age': age,
        'TSH': tsh,
        'T3': t3,
        'T4': t4,
        'Label': target
    })
    return df

# 2. Train the Random Forest Model
def train_pipeline():
    print("Generating laboratory data...")
    df = generate_mock_data()
    
    X = df[['Age', 'TSH', 'T3', 'T4']]
    y = df['Label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model Training Complete. Baseline Accuracy: {acc * 100:.2f}%")
    
    # Save the model
    os.makedirs('model', exist_ok=True)
    with open('model/thyroid_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model successfully saved to 'model/thyroid_model.pkl'")

if __name__ == "__main__":
    train_pipeline()
