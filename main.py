from fastapi import FastAPI

app= FastAPI()
@app.get("/")
async def home_root():
    return {"message" :"success"}

@app.get("/deploy")
async def deploy_root():
    return {"message": "Vercel Deployments"}
