import numpy as np
import pandas as pd
import dill
from fastapi import FastAPI, File
from fastapi.responses import JSONResponse

from transformers import TransformerFechas, TransformerDistancia, TransformerVelocidad

app = FastAPI(title="Taxi Trips Duration Predictor")

# import the preprocessor
with open("preprocesser.pkl", "rb") as f:
    preprocessor = dill.load(f)
# import the model
with open("lr_model.pkl", "rb") as f:
    model = dill.load(f)


@app.get("/", response_class=JSONResponse)
def get_funct(vendor_id: int,
              pickup_datetime: str,
              passenger_count: int,
              pickup_longitude: float,
              pickup_latitude: float,
              dropoff_longitude: float,
              dropoff_latitude: float,
              pickup_borough: str,
              dropoff_borough: str,
              ):
    """Serves predictions given query parameters specifying the taxi trip's
    features from a single example.

    Args:
        vendor_id (int): a code indicating the provider associated with the trip record
        pickup_datetime (str): date and time when the meter was engaged
        passenger_count (float): the number of passengers in the vehicle
        (driver entered value)
        pickup_longitude (float): the longitude where the meter was engaged
        pickup_latitude (float): the latitude where the meter was engaged
        dropoff_longitude (float): the longitude where the meter was disengaged
        dropoff_latitude (float): the latitude where the meter was disengaged
        pickup_borough (str): the borough where the meter was engaged
        dropoff_borough (str): the borough where the meter was disengaged

    Returns:
        [JSON]: model prediction for the single example given
    """
    df = pd.DataFrame(
        [
            [
                vendor_id,
                pickup_datetime,
                passenger_count,
                pickup_longitude,
                pickup_latitude,
                dropoff_longitude,
                dropoff_latitude,
                pickup_borough,
                dropoff_borough,
            ]
        ],
        columns=[
            "vendor_id",
            "pickup_datetime",
            "passenger_count",
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
            "pickup_borough",
            "dropoff_borough",
        ],
    )
    prediction = model.predict(preprocessor.transform(df))
    return {
        "features": {
            "vendor_id": vendor_id,
            "pickup_datetime": pickup_datetime,
            "passenger_count": passenger_count,
            "pickup_longitude": pickup_longitude,
            "pickup_latitude": pickup_latitude,
            "dropoff_longitude": dropoff_longitude,
            "dropoff_latitude": dropoff_latitude,
            "pickup_borough": pickup_borough,
            "dropoff_borough": dropoff_borough,
        },
        "prediction": list(prediction)[0],
    }


if __name__ == "__main__":
    import uvicorn

    # For local development:
    uvicorn.run("main:app", port=3000, reload=True)
