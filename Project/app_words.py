# app.py
import io
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify, render_template
from tensorflow import keras
import keras.backend as K  # ensure Lambda(K.squeeze) can deserialize

# 1) Load model (path must exist)
MODEL_PATH = r"C:/Users/samdc/OneDrive/Desktop/datasets/specialization/root/models/ocr_model.keras"

# If your training used: Lambda(lambda x: K.squeeze(x, 1)), provide a named function
def squeeze_layer(x):
    return K.squeeze(x, 1)

# Load with custom_objects so Lambda + K resolve correctly
act_model = keras.models.load_model(
    MODEL_PATH,
    custom_objects={"K": K, "squeeze_layer": squeeze_layer}
)

# 2) Character set must match training exactly (blank = last index)
CHAR_LIST = list("!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
BLANK_IDX = len(CHAR_LIST)

# 3) Preprocess must mirror training
def process_image_for_model(gray_img):
    # gray_img: numpy uint8 grayscale (H, W)
    w, h = gray_img.shape
    new_w = 32
    new_h = int(h * (new_w / w))
    # OpenCV resize expects (width, height)
    img = cv2.resize(gray_img, (new_h, new_w))
    w, h = img.shape

    if w < 32:
        add_zeros = np.full((32 - w, h), 255, dtype=np.uint8)
        img = np.concatenate((img, add_zeros))
        w, h = img.shape
    if h < 128:
        add_zeros = np.full((w, 128 - h), 255, dtype=np.uint8)
        img = np.concatenate((img, add_zeros), axis=1)
        w, h = img.shape
    if h > 128 or w > 32:
        img = cv2.resize(img, (128, 32))
    img = cv2.subtract(255, img)  # invert like training
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=(0, -1))  # (1, 32, 128, 1)
    return img  # Matches training preprocessing [web:190][web:224]

# 4) Correct CTC greedy decode
def ctc_greedy_decode(preds):
    # preds: (N, T, C)
    input_len = np.full((preds.shape[0],), preds.shape[1], dtype=np.int32)
    dec, _ = keras.backend.ctc_decode(preds, input_length=input_len, greedy=True)
    seq = dec[0].numpy()[0]  # first sample
    return "".join(CHAR_LIST[i] for i in seq if 0 <= i < BLANK_IDX)  # map indices to chars [web:229][web:232]

# 5) Image reader
def read_image_from_request(file_storage):
    image_bytes = file_storage.read()
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("L")  # grayscale
    return np.array(pil_img)

app = Flask(__name__)

@app.route("/", methods=["GET"])
def root_json():
    return jsonify(message="OCR API is running. Open /ui for the web page or POST /predict with 'image'.")

# HTML UI (templates/index.html)
@app.route("/ui", methods=["GET"])
def ui():
    return render_template("index.html")  # Flask loads from templates/ folder [web:239][web:241]

@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok")

# Prediction endpoint (multipart/form-data with field name 'image')
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify(error="no file field 'image'"), 400  # input must be named 'image' [web:224]
    file = request.files["image"]
    if file.filename == "":
        return jsonify(error="empty filename"), 400

    try:
        gray = read_image_from_request(file)
        if gray is None or gray.ndim != 2:
            return jsonify(error="invalid image; ensure a PNG/JPG"), 400
        x = process_image_for_model(gray)
        preds = act_model.predict(x, verbose=0)
        text = ctc_greedy_decode(preds)
        return jsonify(text=text)
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route("/favicon.ico")
def favicon():
    return ("", 204)

if __name__ == "__main__":
    # Run from the same conda env used for training: conda activate dl-gpu; python app.py
    app.run(host="0.0.0.0", port=5000, debug=False)