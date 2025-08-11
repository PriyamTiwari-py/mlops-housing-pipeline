from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import pandas as pd

# Define input schema using Pydantic
class HousingInput(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


app = FastAPI()
model_name = "DecisionTreeRegressor"  # Use the actual model name you registered


@app.on_event("startup")
def load_model():
    global model
    model = mlflow.sklearn.load_model(
        f"models:/{model_name}/1"
    )


@app.get("/")
def home():
    return {"message": "California Housing Price Predictor API is live!"}


@app.post("/predict")
def predict_price(data: HousingInput):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)[0]
    return {"predicted_price": prediction}
