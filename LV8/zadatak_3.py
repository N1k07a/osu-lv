import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

# 1. Učitaj model
model = keras.models.load_model("FNN_MNIST_model.h5")

# 2. Učitaj vlastitu sliku (mora biti u istoj mapi kao skripta)
# Ako je slika u boji, uzimamo samo jedan kanal ili je pretvaramo u sivu
img = mpimg.imread("test_broj.png")

# 3. Prilagodba slike za mrežu
# Ako slika ima RGB, uzmi prosjek ili samo jedan kanal
if len(img.shape) == 3:
    img = img[:,:,0] 

# Re-dimenzioniranje na 28x28 ako već nije
# (ovdje pretpostavljamo da je slika već 28x28, ako nije treba koristiti npr. cv2.resize)
img_ready = img.reshape(1, 28, 28, 1)
img_ready = img_ready.astype("float32") # Skaliranje ako već nije [0,1]

# 4. Klasifikacija
prediction = model.predict(img_ready)
predicted_digit = np.argmax(prediction)

print(f"Mreža kaže da je na slici broj: {predicted_digit}")

plt.imshow(img, cmap='gray')
plt.title(f"Predviđena znamenka: {predicted_digit}")
plt.show()