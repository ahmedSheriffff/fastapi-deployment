import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

class Comment(BaseModel):
    comment: str

class Email(BaseModel):
    email: str

app = FastAPI()

# Load the trained models and vectorizers
vectorizer = joblib.load('vectorizer.pkl')
model = joblib.load('model.pkl')
email_vectorizer = joblib.load('email_vectorizer.pkl')  # Ensure this file is the correct vectorizer for emails
email_model = joblib.load('email_model.pkl')

@app.get("/")
def read_root():
    return {"message": "Welcome to the Fake Comment Detector API"}

@app.post("/detect-fake-comment")
async def detect_fake_comment(comment: Comment):
    try:
        # Preprocess the input comment
        comment_vector = vectorizer.transform([comment.comment])
        prediction = model.predict(comment_vector)[0]
        result = "Fake Comment" if prediction == 1 else "Real Comment"
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect-spam-email")
async def detect_spam_email(email: Email):
    try:
        # Preprocess the input email
        email_vector = email_vectorizer.transform([email.email])
        prediction = email_model.predict(email_vector)[0]
        result = "Spam" if prediction == 1 else "Not Spam"
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
