import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import tensorflow as tf

# === Step 1: Load the Data ===
clinical = pd.read_csv("Esophageal Cancer Clinical Data.csv")
protein = pd.read_csv("Esophageal Cancer Protein Data.csv")

# === Step 2: Merge the Datasets on Patient ID ===
merged = pd.merge(clinical, protein, on="patient_id")
print("Merged dataset shape:", merged.shape)

# === Step 3: Select Features and Labels ===
X_clinical_raw = merged[['gender', 'primary_pathology_radiation_therapy', 'primary_pathology_residual_tumor']]
X_protein_raw = merged.iloc[:, -223:].select_dtypes(include=[np.number])
y = (merged['primary_pathology_residual_tumor'] == 'R1').astype(int)

# === Step 4: Preprocess the Data ===
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), X_clinical_raw.columns)]
)
X_clinical = preprocessor.fit_transform(X_clinical_raw)

scaler = StandardScaler()
X_protein = scaler.fit_transform(X_protein_raw)

# === Step 5: Train-Test Split ===
X_clinical_train, X_clinical_test, X_protein_train, X_protein_test, y_train, y_test = train_test_split(
    X_clinical, X_protein, y, test_size=0.2, random_state=42
)

# === Step 6: Build Multimodal Model ===
# Clinical branch
input_clinical = tf.keras.Input(shape=(X_clinical_train.shape[1],))
x1 = tf.keras.layers.Dense(64, activation='relu')(input_clinical)
x1 = tf.keras.layers.BatchNormalization()(x1)
x1 = tf.keras.layers.Dropout(0.3)(x1)

# Protein branch
input_protein = tf.keras.Input(shape=(X_protein_train.shape[1],))
x2 = tf.keras.layers.Dense(128, activation='relu')(input_protein)
x2 = tf.keras.layers.BatchNormalization()(x2)
x2 = tf.keras.layers.Dropout(0.3)(x2)

# Fusion
combined = tf.keras.layers.concatenate([x1, x2])
x = tf.keras.layers.Dense(128, activation='relu')(combined)
x = tf.keras.layers.Dropout(0.3)(x)
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=[input_clinical, input_protein], outputs=output)

# === Step 7: Compile and Train ===
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    [X_clinical_train, X_protein_train],
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32
)

# === Step 8: Evaluate ===
loss, acc = model.evaluate([X_clinical_test, X_protein_test], y_test)
print(f"Test Accuracy: {acc:.4f}")
