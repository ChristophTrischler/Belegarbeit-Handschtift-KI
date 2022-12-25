from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from src.api.api import router
app = FastAPI()

app.include_router(router, prefix="/api")

app.mount("/", StaticFiles(directory="src/api/html", html=True))
