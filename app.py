import uvicorn
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from PIL import Image, ImageOps
import numpy as np
import io
from tensorflow import keras
from fastapi.encoders import jsonable_encoder

app = FastAPI()

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = keras.models.load_model("models/mobile_net.h5", compile=False)

# Load the labels
class_names = open("models/labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Read and process the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array

        # Predict the image
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        result = { 
                  "predicted_class": class_name.strip(),             
                  "confidence_score": float(confidence_score)
                  }
    except:
        result = {"Error": "Failed to predict image"}

    return jsonable_encoder(result)

if __name__ == "__main__":
    uvicorn.run("app:app", reload=True, access_log=False)
