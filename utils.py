import numpy as np
from keras.preprocessing.image import load_img, img_to_array

def preprocess_image(image_path):
    img = load_img(image_path, color_mode='grayscale', target_size=(48, 48))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0
    img = np.repeat(img, 3, axis=-1)
    return img

def predict_emotion(image_path, model, emotion_labels):
    processed_img = preprocess_image(image_path)
    preds = model.predict(processed_img, verbose=0)
    emotion = emotion_labels[np.argmax(preds)]
    confidence = np.max(preds)
    return emotion, confidence
