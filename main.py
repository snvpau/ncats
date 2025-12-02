import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Cargar los datos 

train_data_flat = np.loadtxt('dataset/train/train_images.csv', delimiter=',')
train_labels = np.loadtxt('dataset/train/train_labels.csv', delimiter=',')

test_data_flat = np.loadtxt('dataset/test/test_images.csv', delimiter=',')
test_labels = np.loadtxt('dataset/test/test_labels.csv', delimiter=',')

train_data = train_data_flat / 255.0
test_data = test_data_flat / 255.0

train_images = train_data.reshape(-1, 64, 64, 3)
test_images = test_data.reshape(-1, 64, 64, 3)

train_labels = train_labels.reshape(-1, 1)
test_labels = test_labels.reshape(-1, 1)

print("Train images shape:", train_images.shape)
print("Train labels shape:", train_labels.shape)
print("Test images shape:", test_images.shape)
print("Test labels shape:", test_labels.shape)

# Definir la CNN

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # salida binaria (0 = no gato, 1 = gato)
])

model.summary()

# Compilar

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Entrenamiento 

EPOCHS = 10
BATCH_SIZE = 32

history = model.fit(
    train_images,
    train_labels,
    validation_data=(test_images, test_labels),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

loss, acc = model.evaluate(test_images, test_labels, verbose=0)
print(f"Accuracy en test: {acc:.4f}")

model.save("cnn_model_ncats.h5")
print("Modelo guardado como cnn_model_ncats.h5")
