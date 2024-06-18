from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from PIL import Image
import io
import requests
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
import math

app = FastAPI()

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        print(f"Request URL: {request.url}")
        print(f"Request Headers: {request.headers}")
        response = await call_next(request)
        return response

app.add_middleware(RequestLoggingMiddleware)

# URLs to the models
MODEL_URL = 'https://storage.googleapis.com/bucket-storage-request/best.pt'
MODEL_2_URL = 'https://storage.googleapis.com/bucket-storage-request/model2.h5'

# Function to download a model
def download_model(url, destination):
    response = requests.get(url)
    with open(destination, 'wb') as f:
        f.write(response.content)

# Download and load the models at startup
model1_path = 'best.pt'
model2_path = 'model2.h5'

download_model(MODEL_URL, model1_path)
download_model(MODEL_2_URL, model2_path)

# Load YOLO model
model1 = YOLO(model1_path, task='detect')  # Define task explicitly

# Load TensorFlow/Keras model with proper custom object handling
@tf.keras.utils.register_keras_serializable()
def mse(y_true, y_pred):
    return MeanSquaredError()(y_true, y_pred)

model2 = tf.keras.models.load_model(model2_path, custom_objects={'mse': mse})

def round_price(price):
  """Rounds a price to the nearest multiple of 500.

  Args:
      price: The price to be rounded (float).

  Returns:
      The rounded price as an integer.
  """

  remainder = price % 500
  if remainder == 0:  # Price is already a multiple of 500
    return int(price)
  elif abs(remainder) <= 250:  # Round to nearest multiple (including 250)
    return int(price - remainder)  # Always round down on exact midpoint
  elif remainder > 250:  # Round up if above midpoint
    return int(price + (500 - remainder))
  else:  # Round down if below midpoint (including negative remainders)
    return int(price - remainder)

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...), purchase_price: float = Form(...)):
    print("Received a file upload request.")
    
    try:
        # Check file details
        print(f"Filename: {file.filename}")
        print(f"Content-Type: {file.content_type}")

        # Read the uploaded image
        contents = await file.read()
        print(f"File contents length: {len(contents)}")

        # Load the image
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Convert image to a numpy array
        img_np = np.array(image)

        # Log image dimensions and type
        print(f"Image shape: {img_np.shape}, Image dtype: {img_np.dtype}")

        # Perform prediction with thresholds
        results = model1(img_np, conf=0.1, iou=0.01)

        # Initialize counters for the areas of ripped and wornout regions
        total_area_ripped = 0
        total_area_wornout = 0

        # Initialize counts for the number of ripped and wornout points
        total_ripped_count = 0
        total_wornout_count = 0

        # Assume there is only one image in the batch
        result = results[0]

        # Print raw results for debugging
        print("Raw results:", result)
        
        # Get image dimensions and calculate the area of the book cover
        array_img = result.orig_shape  # Using orig_shape to get image dimensions
        print("Image dimensions:", array_img)
        luas_buku = array_img[0] * array_img[1]
        print("Book cover area:", luas_buku)
        
        # Extract class and box predictions
        cls_array = result.boxes.cls.cpu().numpy()
        xywh_array = result.boxes.xywh.cpu().numpy()
        print("Class array:", cls_array)
        print("Bounding boxes (xywh):", xywh_array)
        
        # Calculate total area for each class
        for i in range(len(cls_array)):
            width = xywh_array[i][2]
            height = xywh_array[i][3]
            luas_box = width * height
            if cls_array[i] == 0:  # Ripped
                total_area_ripped += luas_box
                total_ripped_count += 1
            elif cls_array[i] == 1:  # Wornout
                total_area_wornout += luas_box
                total_wornout_count += 1

        # Calculate ratios
        rasio_ripped = (total_area_ripped / luas_buku) * 100
        rasio_wornout = (total_area_wornout / luas_buku) * 100

        # Calculate overall ratio
        overall_ratio = (200 - (rasio_ripped + rasio_wornout)) / 2

        # Prepare the result dictionary
        resultDict = {
            'Wornout': total_wornout_count,
            'Ripped': total_ripped_count,
            'Rasio_Ripped': round(rasio_ripped, 2),
            'Rasio_Wornout': round(rasio_wornout, 2),
            'Overall_Ratio': round(overall_ratio, 2)
        }

        # Print detailed results
        print(f"Ditemukan {resultDict['Wornout']} titik wornout dan {resultDict['Ripped']} titik ripped")

        # Print results for debugging
        print("Results:", resultDict)

        # Data to be sent to model 2
        # Adding purchase_price as the first feature
        data_to_send = np.array([[purchase_price, resultDict['Rasio_Ripped'], resultDict['Rasio_Wornout']]])
        
        # Predict with model 2
        try:
            predicted_price = model2.predict(data_to_send) / 100
            predicted_price_float = float(predicted_price[0][0])  # Convert to float for JSON serialization

            # Print raw predicted price
            print("Model 2 predicted price (raw):", predicted_price_float)

            # Round the predicted price
            if predicted_price_float - int(predicted_price_float) >= 0.5:
                predicted_price_rounded = math.ceil(predicted_price_float)
            else:
                predicted_price_rounded = math.floor(predicted_price_float)

            print("Model 2 predicted price (rounded):", predicted_price_rounded)

            final_predict_price = round_price(predicted_price_rounded)

            print("Final predicted price (rounded to nearest 500):", final_predict_price)

            # Combine results from both models
            final_result = {
                **resultDict,
                'Recommended_Price': final_predict_price  # Assuming the output is a single value
            }

            return JSONResponse(content=final_result)
        except Exception as e:
            print("Error occurred while calling model 2:", str(e))
            raise HTTPException(status_code=500, detail=f"Error calling model 2: {str(e)}")

    except Exception as e:
        print("Error occurred:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
