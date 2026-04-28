# ============================================================
# WeldWise — FastAPI Backend  (Deploy on Render.com for free)
# ============================================================
# Folder structure:
#   weldwise-api/
#   ├── app.py              ← this file
#   ├── requirements.txt
#   └── models/
#       ├── model_classifier.pkl
#       ├── label_encoder.pkl
#       ├── model_yield.pkl
#       ├── model_uts.pkl
#       ├── model_elongation.pkl
#       └── features.json
# ============================================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import json
import numpy as np

app = FastAPI(title="WeldWise API")

# Allow your Wix site to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your Wix domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models once at startup ──────────────────────────────
clf  = joblib.load("models/model_classifier.pkl")
le   = joblib.load("models/label_encoder.pkl")
reg_yield = joblib.load("models/model_yield.pkl")
reg_uts   = joblib.load("models/model_uts.pkl")
reg_elon  = joblib.load("models/model_elongation.pkl")

with open("models/features.json") as f:
    FEATURES = json.load(f)

# ── Request schema ───────────────────────────────────────────
class WeldInput(BaseModel):
    material_thickness:   float
    current:              float
    voltage:              float
    weld_speed:           float
    shielding_gas_flow:   float
    filler_wire_diameter: float

# ── Prediction endpoint ──────────────────────────────────────
@app.post("/predict")
def predict(data: WeldInput):
    # Derive heat input
    heat_input = (data.voltage * data.current) / data.weld_speed

    row = np.array([[
        data.material_thickness,
        data.current,
        data.voltage,
        data.weld_speed,
        data.shielding_gas_flow,
        data.filler_wire_diameter,
        heat_input,
    ]])

    condition     = le.inverse_transform(clf.predict(row))[0]
    yield_strength = round(float(reg_yield.predict(row)[0]), 1)
    uts            = round(float(reg_uts.predict(row)[0]), 1)
    elongation     = round(float(reg_elon.predict(row)[0]), 1)

    return {
        "weld_condition":  condition,
        "yield_strength":  yield_strength,
        "uts":             uts,
        "elongation":      elongation,
    }

# Health check
@app.get("/")
def root():
    return {"status": "WeldWise API is running 🔧"}
