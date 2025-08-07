from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np

# Load dataset
digits = load_digits()
X = digits.images / 16.0  # Normalize
X = np.expand_dims(X, -1)  # (n, 8, 8, 1)
y = to_categorical(digits.target, 10)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(8,8,1)),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=5, validation_split=0.1)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {acc:.2f}")

# Explainability (Description)
"""
SHAP and LIME are tools that help explain why a model made a decision.
- SHAP assigns a value to each input (e.g. pixel) showing how it influenced the prediction.
- LIME perturbs input and builds a simple model to explain locally.

This helps detect model bias or mistakes and builds trust in AI systems.
"""