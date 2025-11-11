from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np

train_data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_data_gen.flow_from_directory(
    'data/train',           # folder path with subfolders for each class
    target_size=(48, 48),
    color_mode='grayscale',
    batch_size=64,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_data_gen.flow_from_directory(
    'data/train',
    target_size=(48, 48),
    color_mode='grayscale',
    batch_size=64,
    class_mode='categorical',
    subset='validation'
)



model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')   # 7 emotion classes in FER2013
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=30)


model.save('emotion_model.h5')
