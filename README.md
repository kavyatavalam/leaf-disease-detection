# Leaf Disease Detection System

A machine learning web application that detects plant leaf diseases from uploaded images using a trained deep learning model.

## Features

- User Signup and Login
- Upload plant leaf images
- Predict leaf disease using a CNN model
- Display prediction results with confidence score
- Store and view prediction history

## Technologies Used

### Backend
- Python
- FastAPI
- TensorFlow / Keras
- SQLite

### Frontend
- HTML
- CSS
- JavaScript

### Tools
- Git
- GitHub
- VS Code

## Project Structure

LDD/
├── backend/
├── frontend/
├── training/
├── requirements.txt
├── README.md
└── .gitignore

## Installation and Setup

### 1. Clone the Repository

git clone https://github.com/kavyatavalam/leaf-disease-detection.git
cd leaf-disease-detection

### 2. Create and Activate a Virtual Environment

python -m venv venv
venv\Scripts\activate

### 3. Install Dependencies

pip install -r requirements.txt

### 4. Run the Application

python backend/main.py

### 5. Open in Browser

http://127.0.0.1:8000

## Model Training

python training/train.py

## Future Enhancements

- Support more plant species
- Improve model accuracy
- Deploy to cloud platforms
- Add user dashboard and analytics

## Author

Kavya Tavalam
