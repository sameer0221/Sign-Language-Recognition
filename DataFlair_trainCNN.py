import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import cv2
import itertools
import random
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

train_path = "dataset/train"   # was D:\gesture\train
test_path  = "dataset/test"    # was D:\gesture\test

train_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input
).flow_from_directory(
    directory=train_path,
    target_size=(64, 64),
    class_mode='categorical',
    batch_size=10,
    shuffle=True
)

test_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input
).flow_from_directory(
    directory=test_path,
    target_size=(64, 64),
    class_mode='categorical',
    batch_size=10,
    shuffle=True
)

imgs, labels = next(train_batches)

def plotImages(images_arr):
    fig, axes = plt.subplots(1, min(10, len(images_arr)), figsize=(20, 4))
    if min(10, len(images_arr)) == 1:
        axes = [axes]
    for img, ax in zip(images_arr, axes):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

plotImages(imgs)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPool2D((2,2), strides=2),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    MaxPool2D((2,2), strides=2),
    Conv2D(128, (3,3), activation='relu', padding='valid'),
    MaxPool2D((2,2), strides=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer=SGD(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=5e-4)
early_stop = EarlyStopping(monitor='val_loss', patience=2, mode='auto', restore_best_weights=True)

history = model.fit(
    train_batches,
    epochs=10,
    callbacks=[reduce_lr, early_stop],
    validation_data=test_batches
)

# evaluate on a small batch
imgs_test, labels_test = next(test_batches)
scores = model.evaluate(imgs_test, labels_test, verbose=0)
print(f'{model.metrics_names[0]}: {scores[0]:.4f}; {model.metrics_names[1]}: {scores[1]*100:.2f}%')

# Save model
import os
os.makedirs("models", exist_ok=True)
model.save("models/best_model_dataflair.h5")

# A simple label map matching train folder names "0".."9":
word_dict = {0:'Zero',1:'One',2:'Two',3:'Three',4:'Four',5:'Five',6:'Six',7:'Seven',8:'Eight',9:'Nine'}

predictions = model.predict(imgs_test, verbose=0)
print("Predictions on a small batch:")
print([word_dict[int(np.argmax(p))] for p in predictions])
print("Actual:")
print([word_dict[int(np.argmax(y))] for y in labels_test])

model.summary()
