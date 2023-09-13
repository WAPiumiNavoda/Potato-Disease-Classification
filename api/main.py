from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

# load the tensorflow model
MODEL = tf.keras.models.load_model("../saved_models/1")
CLASS_NAMES = ["Early Blight", "Late Blight" , "Healthy"]



@app.get("/ping")
async  def ping():
    return "Hello, I am alive"

def read_file_ass_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image
    #BytesIO(data)


@app.post("/predict")
async  def predict(
        file: UploadFile = File(...)
):
    image = read_file_ass_image(await file.read())
    img_batch = np.expand_dims(image, 0) #batch image
    predictions = MODEL.predict(img_batch)
    predicted_class =  CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


    # bytes = await file.read()



if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port = 8000)

