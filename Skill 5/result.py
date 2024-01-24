import numpy as np
import pandas as pd
from keras.models import load_model
from keras.src.utils import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical


def get_diagnosis(filename):
    df = pd.read_csv('train.csv')

    # Encode the labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['diagnosis'])
    num_classes = len(np.unique(y))
    y_categorical = to_categorical(y, num_classes=num_classes)

    # Load the saved model
    model = load_model('my_model.keras')

    # Assuming you have a trained model named 'model'
    # Load and preprocess a new image for prediction
    new_image_path = f'static/{filename}'  # Replace with the path to your new image
    new_image = load_img(new_image_path)
    new_image_array = img_to_array(new_image)
    new_image_array = np.expand_dims(new_image_array, axis=0)  # Add an extra dimension for batch size

    # Make a prediction using the trained model
    predictions = model.predict(new_image_array)

    # Convert the predicted probabilities to a class label
    predicted_class = np.argmax(predictions)

    # Decode the class label using the label_encoder
    predicted_diagnosis = label_encoder.classes_[predicted_class]

    return predicted_diagnosis
