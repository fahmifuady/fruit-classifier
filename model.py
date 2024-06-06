import numpy as np
import tensorflow as tf
from keras.utils import load_img, img_to_array

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="static/model/model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Map the index to class names
class_names = ['Apple Braeburn', 'Apple Granny Smith', 'Apricot', 'Avocado', 'Banana',
               'Blueberry', 'Cactus fruit', 'Cantaloupe', 'Cherry', 'Clementine',
               'Corn', 'Cucumber Ripe', 'Grape Blue', 'Kiwi', 'Lemon', 'Limes',
               'Mango', 'Onion White', 'Orange', 'Papaya', 'Passion Fruit', 'Peach',
               'Pear', 'Pepper Green', 'Pepper Red', 'Pineapple', 'Plum', 'Pomegranate',
               'Potato Red', 'Raspberry', 'Strawberry', 'Tomato', 'Watermelon']

def load_and_preprocess_image(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)  # Add batch dimension
    return img_array

def predict(image_path):
    # Preprocess the image
    img_array = load_and_preprocess_image(image_path)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data, axis=1)

    # Return the predicted class name
    return class_names[predicted_class[0]]
