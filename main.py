from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from api import router
app = FastAPI()

app.include_router(router, prefix="/api")

app.mount("/", StaticFiles(directory="html", html=True))
