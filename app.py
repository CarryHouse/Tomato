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
    except Exception as e:
        print(f"Error: {str(e)}")
        result = {"Error": "Failed to predict image"}

    return jsonable_encoder(result)
