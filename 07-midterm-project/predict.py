import pickle
from pathlib import Path
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

project_dir = Path(__file__).resolve().parent
model_path = project_dir / "model.bin"

with open(model_path, "rb") as file:
    dv, model = pickle.load(file)

app = FastAPI(title="Steam Game Recommendation Predictor API")


class GameFeatures(BaseModel):
    weighted_positive_ratio: float = Field(..., example=94.5)
    price_final: float = Field(..., example=19.99)
    rating: str = Field(..., example="Overwhelmingly Positive")
    tags: str = Field(..., example="action")


class PredictionResponse(BaseModel):
    recommendation_probability: float
    recommended: bool


@app.get("/")
def health_check():
    return {"status": "ok", "model": "LogisticRegression(C=0.01)"}


@app.post("/predict", response_model=PredictionResponse)
def predict(game: GameFeatures):
    game_dict = {
        "weighted_positive_ratio": game.weighted_positive_ratio,
        "price_final": game.price_final,
        "rating": game.rating.strip(),
        "tags": game.tags.lower().strip(),
    }

    X = dv.transform([game_dict])
    prob = float(model.predict_proba(X)[0, 1])

    return PredictionResponse(
        recommendation_probability=round(prob, 4),
        recommended=prob >= 0.5,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)