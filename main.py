# pyrefly: ignore [missing-import]
from fastapi import FastAPI
from src.router import app as router
app=FastAPI()
app.include_router(router)
@app.get("/")
def health_check():
    return {"status": "ok"}
# 
