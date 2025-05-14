from fastapi import FastAPI, File, UploadFile
import numpy as np
import tensorflow.lite as tflite
from PIL import Image
import io
import os
import gdown

app = FastAPI()

# Load TFLite model

FILE_ID = "1aqFgeQ120le-yp6zUZKhpoHK6dx8maaR"
OUTPUT_PATH = "skin_model_v2.tflite"

# Function to download the model
def download_model():
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    if not os.path.exists(OUTPUT_PATH):  # Avoid re-downloading
        print("Downloading the model...")
        gdown.download(url, OUTPUT_PATH, quiet=False)
    else:
        print("Model already exists.")

# Download the model when the server starts
download_model()


interpreter = tflite.Interpreter(model_path="skin_model_v2.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


@app.get('/')
def home():
    return {"message": "Hello Ayurify User!"}

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0 
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    input_data = preprocess_image(image)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    return {"prediction": output_data.tolist()}



if __name__ == "__main__":
    app.run(debug=True)