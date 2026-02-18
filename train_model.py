import os
import glob
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical


# 📁 PATH SETUP (bulletproof)
BASE_DIR = os.path.dirname(__file__)

DATASET_PATH = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(MODEL_DIR, exist_ok=True)


def train():
    # 🔹 Load all CSVs
    files = glob.glob(os.path.join(DATASET_PATH, "*.csv"))

    data = []

    for file in files:
        df = pd.read_csv(file, header=None)
        data.append(df)

    data = pd.concat(data, ignore_index=True)

    # 🔹 Split features and labels
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # 🔹 Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    # 🔹 Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 🔹 Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42
    )

    num_classes = y_categorical.shape[1]

    # 🔹 Build ANN
    model = Sequential([
        Dense(128, activation="relu", input_shape=(63,)),
        Dropout(0.3),

        Dense(64, activation="relu"),
        Dropout(0.3),

        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 🔹 Train
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test)
    )

    # 🔹 Evaluate
    loss, acc = model.evaluate(X_test, y_test)
    print(f"\n✅ Test Accuracy: {acc * 100:.2f}%")

    # 🔹 Save everything
    model.save(os.path.join(MODEL_DIR, "gesture_model.h5"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "labels.pkl"))


    print("🎉 Model saved!")
if __name__ == "__main__":
    train()
