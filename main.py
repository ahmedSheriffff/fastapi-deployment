import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
import os
import logging
import re
import secrets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from datetime import datetime, timedelta, timezone
from typing import Annotated
import jwt
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jwt.exceptions import InvalidTokenError
from passlib.context import CryptContext
from pydantic import BaseModel
import pandas as pd
from passlib.context import CryptContext


# to get a string like this run:
# openssl rand -hex 32
SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


fake_users_db = {
    "johndoe": {
        "username": "johndoe",
        "full_name": "John Doe",
        "email": "johndoe@example.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",
        "disabled": False,
    }
}


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str | None = None


class User(BaseModel):
    username: str
    email: str | None = None
    full_name: str | None = None
    disabled: bool | None = None


class UserInDB(User):
    hashed_password: str


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI()


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)


def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except InvalidTokenError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)],
):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


@app.post("/token")
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> Token:
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")


@app.get("/users/me/", response_model=User)
async def read_users_me(
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    return current_user


@app.get("/users/me/items/")
async def read_own_items(
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    return [{"item_id": "Foo", "owner": current_user.username}]


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




# Function to preprocess text
def preprocess_text(text):
    return re.sub(r"[^\w\s]", "", str(text).lower().strip())

# Load dataset
DATASET_PATH = "review_dataset.csv"
dataset = pd.read_csv(DATASET_PATH)

# Remove rows with NaN values in the 'review' column
dataset = dataset.dropna(subset=["review"])

# Ensure all values are strings and replace any remaining NaN
dataset["review"] = dataset["review"].fillna("").astype(str)

# Apply text preprocessing
dataset["review"] = dataset["review"].apply(preprocess_text)

# Save cleaned dataset
dataset.to_csv("cleaned_review_dataset.csv", index=False)

print("Dataset cleaned successfully!")

# Initialize FastAPI application with middleware
app = FastAPI()

# Paths to the pre-trained model and vectorizer
COMMENT_MODEL_PATH = "fake_comment_model.pkl"
COMMENT_VECTORIZER_PATH = "vectorizer.pkl"
SPAM_MODEL_PATH = "emailmodel.pkl"
SPAM_VECTORIZER_PATH = "spam_vectorizer.pkl"
DATASET_PATH = "fake_comments_dataset.csv"
SPAM_DATASET_PATH = "spam_dataset.csv"
REVIEW_MODEL_PATH = "reviewmodel.pkl"
REVIEW_VECTORIZER_PATH = "review_vectorizer.pkl"
REVIEW_DATASET_PATH = "review_dataset.csv"


# Function to safely load models and vectorizers
def load_model_and_vectorizer(model_path, vectorizer_path, model_name):
    if not os.path.exists(model_path):
        logger.error(f"{model_name} Model file not found: {model_path}")
        raise FileNotFoundError(f"{model_name} Model file not found: {model_path}")
    if not os.path.exists(vectorizer_path):
        logger.error(f"{model_name} Vectorizer file not found: {vectorizer_path}")
        raise FileNotFoundError(f"{model_name} Vectorizer file not found: {vectorizer_path}")
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        logger.info(f"{model_name} model and vectorizer loaded successfully.")
        return model, vectorizer
    except Exception as e:
        logger.error(f"Error loading {model_name} model or vectorizer: {str(e)}")
        raise RuntimeError(f"Failed to load {model_name} model/vectorizer due to: {str(e)}")

# Load models and vectorizers safely
comment_model, comment_vectorizer = load_model_and_vectorizer(COMMENT_MODEL_PATH, COMMENT_VECTORIZER_PATH, "Comment Detection")

# Ensure spam vectorizer exists, otherwise train a new one
if not os.path.exists(SPAM_VECTORIZER_PATH):
    try:
        logger.warning("Spam vectorizer not found. Training a new one.")
        df = pd.read_csv(SPAM_DATASET_PATH)
        df.columns = df.columns.str.strip().str.lower()
        if "email_content" not in df.columns:
            raise ValueError(f"The dataset must contain an 'email_content' column. Found columns: {df.columns}")
        
        vectorizer = TfidfVectorizer(max_features=5000)
        vectorizer.fit(df["email_content"])
        print("Feature Names:", vectorizer.get_feature_names_out())
        print("Number of Features:", len(vectorizer.get_feature_names_out()))
        joblib.dump(vectorizer, SPAM_VECTORIZER_PATH)
        logger.info("Spam vectorizer trained and saved successfully.")
    except Exception as e:
        logger.error(f"Error training spam vectorizer: {str(e)}")
        raise RuntimeError(f"Failed to train spam vectorizer: {str(e)}")


def retrain_vectorizer():
    if not os.path.exists(REVIEW_DATASET_PATH):
        logger.error("Review dataset missing. Cannot retrain vectorizer.")
        raise FileNotFoundError("Review dataset is required for retraining.")

    dataset = pd.read_csv(REVIEW_DATASET_PATH)

    vectorizer = TfidfVectorizer(max_features=1768)
    vectorizer.fit(dataset["review"])
    joblib.dump(vectorizer, REVIEW_VECTORIZER_PATH)

    logger.info(f"Vectorizer retrained with {len(vectorizer.get_feature_names_out())} features.")

# Load models and vectorizer
def load_model_and_vectorizer(model_path, vectorizer_path, model_name):
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        logger.warning(f"{model_name} vectorizer missing, attempting retraining.")
        retrain_vectorizer()

    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)

        # Verify alignment
        sample_text = ["Test review"]
        test_vectorized = vectorizer.transform(sample_text)
        expected_features = 1768

        if test_vectorized.shape[1] != expected_features:
            logger.error(f"{model_name} feature mismatch: Found {test_vectorized.shape[1]}, expected {expected_features}")
            retrain_vectorizer()
            vectorizer = joblib.load(vectorizer_path)

        return model, vectorizer

    except Exception as e:
        logger.error(f"Error loading {model_name}: {str(e)}")
        raise RuntimeError(f"Failed to load {model_name}: {str(e)}")



# Load spam model and vectorizer after ensuring vectorizer exists
spam_model, spam_vectorizer = load_model_and_vectorizer(SPAM_MODEL_PATH, SPAM_VECTORIZER_PATH, "Spam Detection")
review_model, review_vectorizer = load_model_and_vectorizer(REVIEW_MODEL_PATH, REVIEW_VECTORIZER_PATH, "Review Detection")


# Debugging: Check vectorizer feature count
print("Vectorizer feature count:", len(spam_vectorizer.get_feature_names_out()))

# Request and Response Models
class CommentRequest(BaseModel):
    comment: str

class CommentResponse(BaseModel):
    comment: str
    is_fake: bool

class EmailRequest(BaseModel):
    content: str

class EmailResponse(BaseModel):
    content: str
    is_spam: bool

class CommentLabelRequest(BaseModel):
    comment: str
    label: int
    
class ReviewRequest(BaseModel):
    review: str

class ReviewResponse(BaseModel):
    review: str
    is_fake: bool

@app.get("/")
def root():
    return{"message":"welcome to the tustify API"}


@app.post("/detect_fake_comment", response_model=CommentResponse)
async def detect_fake_comment(request: CommentRequest):
    try:
        if not request.comment.strip():
            logger.warning("Empty comment received")
            raise HTTPException(status_code=400, detail="Comment cannot be empty.")
        transformed_comment = comment_vectorizer.transform([request.comment])
        prediction = comment_model.predict(transformed_comment)
        result = bool(prediction[0])
        logger.info(f"Comment: {request.comment} -> Predicted: {result}")
        return CommentResponse(comment=request.comment, is_fake=result)
    except Exception as e:
        logger.error(f"Error processing comment: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

@app.post("/detect_spam_email", response_model=EmailResponse)
async def detect_spam_email(request: EmailRequest):
    try:
        if not request.content.strip():
            logger.warning("Empty email content received.")
            raise HTTPException(status_code=400, detail="Email content cannot be empty.")
        
        if len(request.content.strip()) < 3:
            logger.warning("Short email content received, defaulting to not spam.")
            return EmailResponse(content=request.content, is_spam=False)
        
        processed_content = re.sub(r"[^\w\s]", "", request.content.lower())
        transformed_content = spam_vectorizer.transform([processed_content])
        
        expected_features = 5000
        if transformed_content.shape[1] == expected_features:
            logger.error(f"Feature mismatch: Expected {expected_features}, but got {transformed_content.shape[1]}")
            raise ValueError(f"Feature mismatch: Expected {expected_features}, but got {transformed_content.shape[1]}")
        
        prediction_proba = spam_model.predict_proba(transformed_content)[0]
        is_spam = prediction_proba[1] > 0.5  # Set threshold at 50%
        
        logger.info(f"Email: {request.content} -> Predicted: {'spam' if is_spam else 'not spam'} with probability {prediction_proba[1]:.4f}")
        return EmailResponse(content=request.content, is_spam=is_spam)
    except Exception as e:
        logger.error(f"Error processing email: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@app.post("/detect_fake_review", response_model=ReviewResponse)
async def detect_fake_review(request: ReviewRequest):
    try:
        if not request.review.strip():
            raise HTTPException(status_code=400, detail="Review cannot be empty.")

        # Transform the review
        transformed_review = review_vectorizer.transform([request.review])
        logger.info(f"Transformed review feature count: {transformed_review.shape[1]}")

        # Validate feature count
        expected_features = 1768
        if transformed_review.shape[1] != expected_features:
            logger.error(f"Feature mismatch detected: Expected {expected_features}, Found {transformed_review.shape[1]}")
            retrain_vectorizer()
            transformed_review = review_vectorizer.transform([request.review])

        # Predict with probability check
        prediction_proba = review_model.predict_proba(transformed_review)[0]
        is_fake = prediction_proba[1] > 0.5  # Threshold at 50%

        logger.info(f"Review: {request.review} -> Predicted: {'Fake' if is_fake else 'Real'} with probability {prediction_proba[1]:.4f}")
        return ReviewResponse(review=request.review, is_fake=is_fake)

    except Exception as e:
        logger.error(f"Error processing review: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@app.post("/add_comment")
async def add_comment(request: CommentLabelRequest):
    try:
        if not request.comment.strip():
            logger.warning("Empty comment received for addition")
            raise HTTPException(status_code=400, detail="Comment cannot be empty.")
        new_data = pd.DataFrame({"comment": [request.comment], "label": [request.label]})
        if os.path.exists(DATASET_PATH):
            df = pd.read_csv(DATASET_PATH)
            df = pd.concat([df, new_data], ignore_index=True)
        else:
            df = new_data
        df.to_csv(DATASET_PATH, index=False)
        logger.info(f"Added new comment: {request.comment}")
        return {"message": "Comment added successfully."}
    except Exception as e:
        logger.error(f"Error adding comment: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")
    
@app.post("/add_fake_review")
async def add_fake_review(request: CommentLabelRequest):
    try:
        if not request.comment.strip():
            logger.warning("Empty comment received for addition")
            raise HTTPException(status_code=400, detail="Comment cannot be empty.")
        if request.label != 1:  # Ensure the label is set as "1" for fake
            logger.warning("Invalid label for fake review. Expected label '1'")
            raise HTTPException(status_code=400, detail="Label for fake reviews must be '1'.")
        new_data = pd.DataFrame({"comment": [request.comment], "label": [request.label]})
        if os.path.exists(REVIEW_DATASET_PATH):
            df = pd.read_csv(REVIEW_DATASET_PATH)
            df = pd.concat([df, new_data], ignore_index=True)
        else:
            df = new_data
        df.to_csv(REVIEW_DATASET_PATH, index=False)
        logger.info(f"Added new fake review: {request.comment}")
        return {"message": "Fake review added successfully."}
    except Exception as e:
        logger.error(f"Error adding fake review: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")