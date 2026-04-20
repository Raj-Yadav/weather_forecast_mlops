import os
import pickle
import sqlite3
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- LOAD MODEL ON STARTUP ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model_.pkl")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model once at startup and store on app state
    with open(MODEL_PATH, "rb") as f:
        _model = pickle.load(f)
    app.state.coefficients = _model["coefficients"]
    app.state.intercept    = _model["intercept"]
    yield
    # (teardown logic here if needed)

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- DATABASE HELPER ---
def get_db() -> sqlite3.Connection:
    """Open a SQLite connection and ensure the users table exists."""
    db = sqlite3.connect("users.db")
    db.execute(
        "CREATE TABLE IF NOT EXISTS users "
        "(id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT)"
    )
    db.commit()
    return db


# --- PYDANTIC SCHEMAS ---
class PredictRequest(BaseModel):
    humidity:           float
    wind_speed:         float
    condition_encoded:  int

class AuthRequest(BaseModel):
    username: str
    password: str


# --- ENDPOINTS ---

@app.get("/health")
def health(request: Request):
    """Simple health check — returns 200 when the server is running."""
    return {"status": "healthy", "model": "loaded"}


@app.post("/predict")
def predict(body: PredictRequest, request: Request):
    """
    Predict temperature from normalized weather features.

    Inputs must be normalized using the same mean/std values
    computed during training. See preprocessing.py for reference.
    """
    X          = np.array([[body.humidity, body.wind_speed, body.condition_encoded]])
    prediction = float(X @ request.app.state.coefficients + request.app.state.intercept)
    return {"predicted_temperature": round(prediction, 2)}


@app.post("/register", status_code=201)
def register(body: AuthRequest):
    """Register a new user with username and password."""
    db = get_db()
    try:
        db.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (body.username, body.password),
        )
        db.commit()
        return {"message": "User registered successfully"}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Username already exists")
    finally:
        db.close()


@app.post("/login")
def login(body: AuthRequest):
    """Authenticate a user."""
    db  = get_db()
    row = db.execute(
        "SELECT id FROM users WHERE username=? AND password=?",
        (body.username, body.password),
    ).fetchone()
    db.close()
    if row:
        return {"message": "Login successful", "user_id": row[0]}
    raise HTTPException(status_code=401, detail="Invalid credentials")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=False)