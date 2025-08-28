from flask import Flask, render_template, Response, request
import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from deepface import DeepFace
import datetime

app = Flask(__name__)

# ================================
# MODEL LOADING (h5 pretrained)
# ================================
deepfake_model = load_model("deepfake_detection_model.keras")

# ================================
# CAMERA SETUP
# ================================
camera = cv2.VideoCapture(0)

# ================================
# CNN-LSTM VIDEO FRAME PREP
# ================================
def prepare_frames_for_deepfake(video_path, num_frames=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // num_frames, 1)
    selected_frames = [i for i in range(0, total_frames, step)][:num_frames]

    for i in selected_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (299, 299))  # Match model input size
        frame = frame.astype('float32') / 255.0
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise ValueError("No frames extracted from video")

    frames = np.array(frames)
    frames = np.expand_dims(frames, axis=0)  # Shape: (1, num_frames, 299, 299, 3)
    return frames

def predict_deepfake_video(video_path):
    frames = prepare_frames_for_deepfake(video_path)
    prediction = deepfake_model.predict(frames)
    return prediction[0][0] * 100  # Return percentage confidence

# ================================
# ROUTES
# ================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files and 'video' not in request.files:
        return "No file uploaded", 400

    emotion_data = {}
    deepfake_message = ""

    if 'image' in request.files:
        image = request.files['image']
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = os.path.join('static', 'uploaded_images', f"{timestamp}.jpg")
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        image.save(img_path)

        # Emotion detection
        try:
            result = DeepFace.analyze(img_path=img_path, actions=['emotion'], enforce_detection=False)
            emotion_data = result[0]['emotion']
            print("Emotion detection:", emotion_data)
        except Exception as e:
            print("Emotion detection error:", str(e))

        deepfake_message = "⚠️ Deepfake model supports only videos. Please upload a video."

        return render_template('result.html', result_img=img_path, predictions=emotion_data, deepfake_message=deepfake_message)

    elif 'video' in request.files:
        video = request.files['video']
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join('static', 'uploaded_videos', f"{timestamp}.mp4")
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        video.save(video_path)

        # Emotion detection from first frame
        try:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                emotion_data = result[0]['emotion']
                print("Emotion detection:", emotion_data)
        except Exception as e:
            print("Emotion detection error:", str(e))

        # Deepfake detection
        try:
            deepfake_confidence = predict_deepfake_video(video_path)
            if deepfake_confidence > 50:
                deepfake_message = f"⚠️ Deepfake Detected (Confidence: {deepfake_confidence:.2f}%)"
            else:
                deepfake_message = f"✅ No Deepfake Detected (Confidence: {100 - deepfake_confidence:.2f}%)"
            print("Deepfake confidence score:", deepfake_confidence)
        except Exception as e:
            print("Deepfake detection error:", str(e))
            deepfake_message = "Deepfake detection failed."

        return render_template('result.html', result_video=video_path, predictions=emotion_data, deepfake_message=deepfake_message)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/how')
def how():
    return render_template('how.html')

if __name__ == '__main__':
    app.run(debug=True)
