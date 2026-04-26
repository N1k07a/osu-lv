import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# train i test podaci
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# prikaz karakteristika train i test podataka
print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

# TODO: prikazi nekoliko slika iz train skupa


# skaliranje slike na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# slike trebaju biti (28, 28, 1)
x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)

print("x_train shape:", x_train_s.shape)
print(x_train_s.shape[0], "train samples")
print(x_test_s.shape[0], "test samples")


# pretvori labele
y_train_s = keras.utils.to_categorical(y_train, num_classes)
y_test_s = keras.utils.to_categorical(y_test, num_classes)


# 2. TODO: kreiraj model pomocu keras.Sequential()
model = keras.Sequential([
    keras.Input(shape=input_shape),
    layers.Flatten(),              # Prebacuje 28x28 sliku u dugački niz od 784 broja
    layers.Dense(512, activation="relu"), # Prvi skriveni sloj
    layers.Dense(256, activation="relu"), # Drugi skriveni sloj
    layers.Dense(num_classes, activation="softmax") # Izlazni sloj (vjerojatnost za 10 znamenki)
])
model.summary()

# 3. TODO: definiraj karakteristike procesa ucenja pomocu .compile()
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# 4. TODO: provedi ucenje mreze
batch_size = 128
epochs = 10 # Možeš staviti i manje (npr. 5) za brži test
model.fit(x_train_s, y_train_s, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# 5. TODO: Prikazi test accuracy i matricu zabune
score = model.evaluate(x_test_s, y_test_s, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Predikcija i matrica zabune
y_pred_s = model.predict(x_test_s)
y_pred = np.argmax(y_pred_s, axis=1) # Vrati iz one-hot u obične brojeve (0-9)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.show()

# 6. TODO: spremi model
model.save("FNN_MNIST_model.h5")
print("Model spremljen kao FNN_MNIST_model.h5")