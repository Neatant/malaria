from fastapi import FastAPI, UploadFile
from util import predict

# Initializing App
app = FastAPI()

# Allowing Cross Origins
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Schema For Request
from pydantic import BaseModel
class Schema(BaseModel):
    image: UploadFile

print(" ........... App Started ........... ")

# Endpoints

@app.get("/")
def index():
    return "This is the API for Malaria Detection CNN Model"

@app.post("/predict")
async def endpoint_malaria_detection(image: UploadFile):
    prediction = predict(await image.read())
    return prediction