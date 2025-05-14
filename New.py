import pandas as pd
import numpy as np
import os
from PIL import Image
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
import tensorflow as tf
import matplotlib.pyplot as plt

# Load and merge data
clinical = pd.read_csv("Lung Cancer Clinical Data.csv")
protein = pd.read_csv("Lung Cancer Protein Data.csv")
merged = pd.merge(clinical, protein, on="patient_id")
filtered = merged[merged['primary_pathology_residual_tumor'].isin(['R0', 'R1'])].copy()

# Create balanced fake labels
num_samples = min(len(filtered), len(os.listdir("images")))
y = np.array([0, 1] * (num_samples // 2))[:num_samples]
np.random.shuffle(y)

# Preprocess clinical and protein data
X_clinical_raw = filtered[['gender', 'primary_pathology_radiation_therapy']][:num_samples]
X_protein_raw = filtered.select_dtypes(include=[np.number]).iloc[:, -223:][:num_samples]
preprocessor = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), X_clinical_raw.columns)])
X_clinical = preprocessor.fit_transform(X_clinical_raw)
X_protein = StandardScaler().fit_transform(X_protein_raw.fillna(0))

# Preprocess images
image_dir = "images"
image_size = (224, 224)
image_array = []
for file in sorted(os.listdir(image_dir)):
    if file.endswith(".jpeg") and len(image_array) < num_samples:
        try:
            img = Image.open(os.path.join(image_dir, file)).convert("RGB").resize(image_size)
            image_array.append(np.array(img) / 255.0)
        except:
            continue
X_image = np.array(image_array)

# Train-test split
Xc_train, Xc_test, Xp_train, Xp_test, Xi_train, Xi_test, y_train, y_test = train_test_split(
    X_clinical, X_protein, X_image, y, test_size=0.2, random_state=42
)

# Model architecture
input_clinical = tf.keras.Input(shape=(Xc_train.shape[1],))
x1 = tf.keras.layers.Dense(128, activation='relu')(input_clinical)
x1 = tf.keras.layers.Dense(64, activation='relu')(x1)
x1 = tf.keras.layers.Dense(32, activation='relu')(x1)

input_protein = tf.keras.Input(shape=(Xp_train.shape[1],))
x2 = tf.keras.layers.Dense(256, activation='relu')(input_protein)
x2 = tf.keras.layers.Dense(128, activation='relu')(x2)
x2 = tf.keras.layers.Dropout(0.3)(x2)
x2 = tf.keras.layers.Dense(64, activation='relu')(x2)

input_image = tf.keras.Input(shape=(224, 224, 3))
base_model = tf.keras.applications.MobileNetV2(include_top=False, input_shape=(224, 224, 3), pooling='avg', weights='imagenet')
base_model.trainable = False
x3 = base_model(input_image)
x3 = tf.keras.layers.Dense(128, activation='relu')(x3)
x3 = tf.keras.layers.Dense(64, activation='relu')(x3)

# Combine all three
combined = tf.keras.layers.concatenate([x1, x2, x3])
z = tf.keras.layers.Dense(128, activation='relu')(combined)
z = tf.keras.layers.Dropout(0.3)(z)
z = tf.keras.layers.Dense(64, activation='relu')(z)
output = tf.keras.layers.Dense(1, activation='sigmoid')(z)

model = tf.keras.Model(inputs=[input_clinical, input_protein, input_image], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])

# Train model (100 epochs)
history = model.fit([Xc_train, Xp_train, Xi_train], y_train, validation_split=0.2, epochs=100, batch_size=8)

# Predict
y_pred_probs = model.predict([Xc_test, Xp_test, Xi_test])
y_pred_probs = np.clip(y_pred_probs, 0, 1)
y_pred = (y_pred_probs > 0.5).astype(int)

# Save Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix_3input.png")

# Save Accuracy Plot
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("Accuracy Over Epochs")
plt.savefig("accuracy_3input.png")

# Save Loss Plot
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Loss Over Epochs")
plt.savefig("loss_3input.png")

# AUC Score
try:
    from sklearn.utils import column_or_1d
    y_test = column_or_1d(y_test)
    auc = roc_auc_score(y_test, y_pred_probs)
    print(f"AUC Score: {auc:.4f}")
except ValueError as e:
    print(f"AUC could not be calculated: {e}")
