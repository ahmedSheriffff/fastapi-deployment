import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def home():
    return {"message": "Success"}
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)