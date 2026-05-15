from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from models import User
from database import SessionLocal
from passlib.context import CryptContext
from pydantic import BaseModel

router = APIRouter()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ✅ Request body model
class UserAuth(BaseModel):
    username: str
    password: str

# DB connection
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# SIGNUP
@router.post("/signup")
def signup(user: UserAuth, db: Session = Depends(get_db)):
    hashed_password = pwd_context.hash(user.password)

    new_user = User(username=user.username, password=hashed_password)

    try:
        db.add(new_user)
        db.commit()
        return {"message": "User created"}
    except:
        return {"error": "Username already exists"}

# LOGIN
@router.post("/login")
def login(user: UserAuth, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()

    if db_user and pwd_context.verify(user.password, db_user.password):
        return {"message": "Login success"}

    return {"error": "Invalid credentials"}