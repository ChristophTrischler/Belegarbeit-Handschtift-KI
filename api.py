import keras.models
from fastapi import FastAPI, UploadFile, File, APIRouter
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uuid
import numpy as np
import cv2
from datetime import datetime
from image import *
from model import *

router = APIRouter()


model = keras.models.load_model("model2")


@router.get("/images/{name}")
async def getImg(name: str):
    return FileResponse(f"images/{name}")


@router.post("/image")
async def postImage(file: UploadFile = File(...)):
    # loads img
    content = await file.read()
    nparr = np.fromstring(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # saves the img with current datetime as name
    filename = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
    img, nums = readTest(img)
    cv2.imwrite(f"images/{filename}.png", img)

    for n in nums:
        predictions = testImgs(n, model)  # predictions of the numbers and accuracy
        print(predictions)

    return {
        "filename": f"{filename}",  
        "nums": [str(p) for p in predictions],  # list of np.ints to normal ints
        "accuracy": 1,
    }


@router.get("/")
async def root():
    return {"test": True}
