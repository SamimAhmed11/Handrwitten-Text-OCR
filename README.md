 Handwritten OCR (Flask + TensorFlow)

A simple Handwritten Text Recognition (HTR) system built with TensorFlow / Keras, OpenCV, and a Flask API. 
It performs OCR on line-level handwritten text images (similar to the IAM dataset). 
The project includes preprocessing, CTC decoding, and a web UI for uploading images.

 Features
-  CRNN + CTC based handwriting recognition
-  Flask REST API
-  Upload image → get recognized text
-  Preprocessing (resize → pad → invert → normalize)
-  Greedy CTC decode
-  Simple HTML UI at /ui
-  Works inside a Conda environment

 Project Structure
app_words.py
models/ocr_model.keras
templates/index.html
static/
data/

 Model Used
CTC-based CRNN model saved as:
- ocr_model.keras 
The model file is NOT included. You must download or train your own.

 Conda Environment
conda create -n ocr-env python=3.10
conda activate ocr-env
pip install tensorflow numpy pandas opencv-python Pillow matplotlib flask

 Running the Application
python app_words.py
Open: http://localhost:5000/ui

 API Usage
POST /predict:
curl -X POST -F "image=@line.png" http://localhost:5000/predict

 Sample Dataset (IAM-like)
Text lines like: “A MOVE to stop Mr. Gaitskell...”

 How To Get the Model
Download your pretrained model OR train your own using IAM lines, CHAR_LIST, CTC.

 Troubleshooting
Empty outputs = preprocessing mismatch.
Model load errors = Keras version or Lambda layer missing custom_objects.

 License
Educational and research use only.



