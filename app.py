from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2, tempfile, os

app = Flask(__name__)
model = tf.keras.models.load_model('fslsavedmodel_alphabet.tf')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file provided'}), 400

    filename = file.filename.lower()
    if filename.endswith(('.png', '.jpg', '.jpeg')): 
        # Handle image input
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        input_seq = np.expand_dims(img, axis=0)  # shape (1, H, W, C)
        preds = model.predict(input_seq)
    else:
        # Handle video input: save temporarily and extract frames
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=filename)
        file.save(temp_file.name)
        cap = cv2.VideoCapture(temp_file.name)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        cap.release()
        os.unlink(temp_file.name)
        if len(frames) == 0:
            return jsonify({'error': 'Empty video'}), 400
        seq = np.array(frames)
        seq = np.expand_dims(seq, axis=0)  # shape (1, time, H, W, C)
        preds = model.predict(seq)

    class_id = int(np.argmax(preds, axis=1)[0])
    return jsonify({'predicted_class': class_id})
