import uvicorn
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from PIL import Image, ImageOps
import numpy as np
import io
from tensorflow import keras
from fastapi.encoders import jsonable_encoder
import os 

app = FastAPI()

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

cwd = os.getcwd() 
# Get the current working directory 
model_path = os.path.join(cwd, "models", "tomatoes.h5")
 # Join it with the relative path of the model file 
labels_path = os.path.join(cwd, "models", "labels.txt") 
# Join it with the relative path of the labels file 
model = keras.models.load_model(model_path, compile=False) 
# Load the model using the absolute path 
class_names = open(labels_path, "r").readlines() # Load the labels using the absolute path

# # Load the model
# model = keras.models.load_model("models/tomatoes.h5", compile=False)

# # Load the labels
# class_names = open("models/labels.txt", "r").readlines()

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
        img_array = tf.expand_dims(normalized_image_array, 0)

        # Predict the image
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence_score = round(100 * np.max(predictions[0]), 2)

        result = { 
            "predicted_class": predicted_class.strip(),             
            "confidence_score": float(confidence_score)
        }
    except:
        result = {"Error": "Failed to predict image"}

    return jsonable_encoder(result)

if __name__ == "__main__": 
    port = int(os.environ.get("PORT", 8000)) 
    uvicorn.run("app:app", host="0.0.0.0", port=port)
