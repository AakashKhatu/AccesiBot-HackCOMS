from together import Together

client = Together()
from fastapi import FastAPI
from pydantic import BaseModel
from loguru import logger
from fastapi.middleware.cors import CORSMiddleware
import json
from notebooks.FinalChain import main

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


class ImageURL(BaseModel):
    url: str


sampleresponse = ""


@app.post("/analyze")
def index(request: ImageURL):
    global sampleresponse
    logger.info("Running Inference")
    response = main(request.url)
    sampleresponse = response
    return {"analysis": sampleresponse}


@app.post("/analyze_test")
def index(request: ImageURL):
    global sampleresponse
    return {"analysis": sampleresponse}


from pyngrok import ngrok

ngrok_tunnel = ngrok.connect(8000)

print(ngrok_tunnel)
import nest_asyncio
import uvicorn

nest_asyncio.apply()
uvicorn.run(app, port=8000)
