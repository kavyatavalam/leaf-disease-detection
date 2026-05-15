from sqlalchemy import Column, Integer, String
from database import Base
from sqlalchemy import Column, Integer, String, Float

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password = Column(String)

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String)
    disease = Column(String)
    confidence = Column(Float)