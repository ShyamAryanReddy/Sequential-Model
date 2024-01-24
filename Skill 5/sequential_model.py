import pandas as pd
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the CSV file
df = pd.read_csv('train.csv')

# Define the path to the folder containing images
image_folder_path = 'gaussian_filtered_images'


# Function to load and preprocess images
def load_and_preprocess_images(image_folder, image_names, target_size=(224, 224)):
    images = []
    for img_name in image_names:
        img_path = os.path.join(image_folder, img_name + '.png')  # Assuming the images are in PNG format
        if os.path.exists(img_path):
            try:
                img = load_img(img_path)
            except OSError:
                print("Error loading image:", img_path)
                continue
            img_array = img_to_array(img)
            images.append(img_array)
    return np.array(images)


# Load and preprocess images
X = load_and_preprocess_images(image_folder_path, df['id_code'])

# Encode the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['diagnosis'])
num_classes = len(np.unique(y))
y_categorical = to_categorical(y, num_classes=num_classes)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Define the Sequential model
model = Sequential()

# Add convolutional and pooling layers
model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

# Add dense layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# Output layer
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model to a file in the same directory as your code
model.save('my_model.keras')
