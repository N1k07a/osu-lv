import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt

# 1. Učitaj podatke i model
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
model = keras.models.load_model("FNN_MNIST_model.h5")

# 2. Pripremi testne podatke (isto kao kod treniranja)
x_test_s = x_test.astype("float32") / 255
x_test_s = np.expand_dims(x_test_s, -1)

# 3. Predviđanje
y_pred_probs = model.predict(x_test_s)
y_pred = np.argmax(y_pred_probs, axis=1)

# 4. Nađi indekse gdje je predikcija kriva
misclassified_indices = np.where(y_pred != y_test)[0]

# 5. Prikaži nekoliko loših primjera
plt.figure(figsize=(10, 4))
for i, idx in enumerate(misclassified_indices[:5]):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_test[idx], cmap='gray')
    plt.title(f"Stvarno: {y_test[idx]}\nPredviđeno: {y_pred[idx]}")
    plt.axis('off')
plt.tight_layout()
plt.show()