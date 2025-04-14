from fastapi import FastAPI, File, UploadFile
import numpy as np
import tensorflow.lite as tflite
from PIL import Image
import io
import os
import gdown

app = FastAPI()

# Load TFLite model

FILE_ID_EYE = "1nvWsYFZ-aHRFEFOp_kMOJE9bUpjHihe3"
OUTPUT_PATH_EYE = "eye_model.tflite"

FILE_ID_SKIN = "1TjkTrHpDJFdvfCortA8w5xjy9K6Xai04"
OUTPUT_PATH_SKIN = "skin_model.tflite"
FILE_ID_EYE = "1nvWsYFZ-aHRFEFOp_kMOJE9bUpjHihe3"
OUTPUT_PATH_EYE = "eye_model.tflite"

FILE_ID_SKIN = "1TjkTrHpDJFdvfCortA8w5xjy9K6Xai04"
OUTPUT_PATH_SKIN = "skin_model.tflite"

# Function to download the model
def download_model():
    url_eye = f"https://drive.google.com/uc?id={FILE_ID_EYE}"
    url_skin = f"https://drive.google.com/uc?id={FILE_ID_SKIN}"
    if not os.path.exists(OUTPUT_PATH_EYE or OUTPUT_PATH_SKIN):  # Avoid re-downloading
        print("Downloading the model...")
        gdown.download(url_eye, OUTPUT_PATH_EYE, quiet=False)
        gdown.download(url_skin, OUTPUT_PATH_SKIN, quiet=False)
    else:
        print("Model already exists.")

# Download the model when the server starts
download_model()


interpreter_eye = tflite.interpreter_eye(model_path="eye_model.tflite")
interpreter_eye.allocate_tensors()
input_details_eye = interpreter_eye.get_input_details()
output_details_eye = interpreter_eye.get_output_details()


interpreter_skin = tflite.Interpreter(model_path="skin_model.tflite")
interpreter_skin.allocate_tensors()
input_details_skin = interpreter_skin.get_input_details()
output_details_skin = interpreter_skin.get_output_details()


@app.get('/')
def home():
    return {"message": "Hello FastAPI"}

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0 
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image

@app.post("/predict-eye")
async def predict_eye(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    input_data = preprocess_image(image)

    # Run inference
    interpreter_eye.set_tensor(input_details_eye[0]['index'], input_data)
    interpreter_eye.invoke()
    output_data = interpreter_eye.get_tensor(output_details_eye[0]['index'])

    return {"prediction": output_data.tolist()}


@app.post("/predict-skin")
async def predict_skin(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    input_data = preprocess_image(image)

    # Run inference
    interpreter_skin.set_tensor(input_details_skin[0]['index'], input_data)
    interpreter_skin.invoke()
    output_data = interpreter_skin.get_tensor(output_details_skin[0]['index'])

    return {"prediction": output_data.tolist()}

if __name__ == "__main__":
    app.run(debug=True)