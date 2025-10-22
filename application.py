from fastapi import FastAPI, Form
from pydantic import BaseModel
import uvicorn
import os
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = FastAPI(title="Student Score Prediction API")


class InputData(BaseModel):
    gender: str
    race_ethnicity: str
    parental_level_of_education: str
    lunch: str
    test_preparation_course: str
    reading_score: float
    writing_score: float


@app.get("/")
def home():
    return {"message": "Welcome to the Student Score Prediction API 🚀"}


@app.post("/predict")
def predict(data: InputData):
    
    custom_data = CustomData(
        gender=data.gender,
        race_ethnicity=data.race_ethnicity,
        parental_level_of_education=data.parental_level_of_education,
        lunch=data.lunch,
        test_preparation_course=data.test_preparation_course,
        reading_score=data.reading_score,
        writing_score=data.writing_score
    )

    pred_df = custom_data.get_data_as_data_frame()

    predict_pipeline = PredictPipeline()
    prediction = predict_pipeline.predict(pred_df)

    return {"predicted_maths_score": float(prediction[0])}


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("application:app", host="0.0.0.0", port=port)
