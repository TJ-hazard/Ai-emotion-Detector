from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np
import cv2
import base64

app = Flask(__name__)
model = load_model('emotion_model.h5')
class_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Utility: preprocess uploaded or captured images
def preprocess_image(image_bytes):
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))
    img = img.reshape(1,48,48,1) / 255.0
    return img

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Webcam capture (base64 string)
    if request.form.get('webcam_image'):
        data_url = request.form['webcam_image']
        header, encoded = data_url.split(',', 1)
        image_bytes = base64.b64decode(encoded)
        img = preprocess_image(image_bytes)
    # File upload
    elif 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', prediction="No file selected.")
        image_bytes = file.read()
        img = preprocess_image(image_bytes)
    else:
        return render_template('index.html', prediction="No file or image data received.")
    prediction = model.predict(img)
    pred_class = class_labels[np.argmax(prediction)]
    return render_template('index.html', prediction=pred_class)

if __name__ == '__main__':
    app.run(debug=True)
