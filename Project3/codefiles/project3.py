import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ShuffleNetV2, ResNet50, MobileNetV2
from tensorflow.keras.applications import shuffleNetV2, ResNet50, MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

base_dir = 'E:/dataset/BIOMEDICAL_IMAGES_2024_KAGGLE/data'
batch_size = 32
img_height, img_width = 224, 224

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)
validation_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

def create_model(base_model, num_classes):
    base_model.trainable = False
    inputs = base_model.input
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

num_classes = len(train_generator.class_indices)

shufflenet = ShuffleNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
mobilenet = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

shufflenet_model = create_model(shufflenet, num_classes)
resnet_model = create_model(resnet, num_classes)
mobilenet_model = create_model(mobilenet, num_classes)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint_shufflenet = ModelCheckpoint('shufflenet_model.h5', save_best_only=True, monitor='val_loss')
checkpoint_resnet = ModelCheckpoint('resnet_model.h5', save_best_only=True, monitor='val_loss')
checkpoint_mobilenet = ModelCheckpoint('mobilenet_model.h5', save_best_only=True, monitor='val_loss')

shufflenet_history = shufflenet_model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    callbacks=[early_stopping, checkpoint_shufflenet]
)

resnet_history = resnet_model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    callbacks=[early_stopping, checkpoint_resnet]
)

mobilenet_history = mobilenet_model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    callbacks=[early_stopping, checkpoint_mobilenet]
)

def plot_history(history, model_name):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.title(f'{model_name} Accuracy')
    plt.savefig(f'{model_name}_accuracy.png')
    plt.clf()

plot_history(shufflenet_history, 'ShuffleNetV2')
plot_history(resnet_history, 'ResNet50')
plot_history(mobilenet_history, 'MobileNetV2')
