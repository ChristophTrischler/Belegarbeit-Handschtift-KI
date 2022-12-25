import keras.models
from fastapi import FastAPI, UploadFile, File, APIRouter
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uuid
import numpy as np
import cv2
from datetime import datetime
from ..image import *
from ..model import *
from ..data import *
import json

router = APIRouter()


model = keras.models.load_model("src/models/model")


@router.get("/images/{name}")
async def getImg(name: str):
    return FileResponse(f"src/images/{name}")


@router.post("/image/{rows}")
async def postImage(rows: int, file: UploadFile = File(...)):
    print("get image...")
    # loads img
    content = await file.read()
    nparr = np.fromstring(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # saves the img with current datetime as name
    filename = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')

    cv2.imwrite(f"src/images/In_{filename}.png", img)

    img, nums = readTest(img, rows)
    cv2.imwrite(f"src/images/{filename}.png", img)

    predictions = [testImgs(n, model) for n in nums]
    compareTest(predictions)

    for p in predictions:
        print(p)

    return {
        "filename": f"{filename}",
        "result": predictions,
    }


@router.get("/res")
async def getRes():
    return  loadRes()


@router.get("/")
async def root():
    return {"test": True}
