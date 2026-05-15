import numpy as np
import os
import pickle

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# ✅ Dataset path
dataset_path = r"D:\Project_4_1\LDD\dataset"

print("🔥 Starting training...")

# ✅ Remove non-image files
valid_extensions = (".jpg", ".jpeg", ".png")

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if not file.lower().endswith(valid_extensions):
            os.remove(os.path.join(root, file))

print("Dataset cleaned!")

# ✅ Data generator
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# ✅ Save labels
class_labels = list(train_data.class_indices.keys())

os.makedirs(r"D:\Project_4_1\LDD\backend\model", exist_ok=True)

with open(r"D:\Project_4_1\LDD\backend\model\labels.pkl", "wb") as f:
    pickle.dump(class_labels, f)

print("Labels saved!")

# ✅ Model
model = Sequential()

model.add(Input(shape=(128,128,3)))

model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(train_data.num_classes, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Training started...")

model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# ✅ Save model
model.save(r"D:\Project_4_1\LDD\backend\model\plant_disease_model_1.h5")

print("✅ Model saved successfully!")