# ⚡ Loan Risk Intelligence Engine

A professional full-stack machine learning web application that evaluates loan default risk and predicts applicant status 💳. It integrates a trained Scikit-Learn Random Forest classification model with a **FastAPI** backend server that simultaneously serves an interactive HTML frontend dashboard and handles real-time prediction API requests ⚡.

---

## 🎯 Project Overview

The goal of this project is to build a robust predictive pipeline capable of estimating loan default probabilities using comprehensive applicant features such as age, annual income, employment length, loan amount, interest rate, credit history, and one-hot encoded categories (home ownership, loan intent, and loan grade) 📊.

Unlike traditional static scripts or basic dashboard tools, this application bridges machine learning and full-stack engineering by serving a modern dark-themed dashboard straight from your local backend server, making it fully ready for GitHub deployment and cloud scaling ☁️.

---

## 📂 Project Architecture

```text
risk-project/
├── app/
│   ├── main.py              # FastAPI server, route handlers & data endpoints
│   └── templates/
│       └── loan_dashboard.html # Modern dark-themed frontend user interface
├── data/                    # Trained models and serialized pickles (.pkl)
├── models/                  # Dataframes and feature columns (.pkl)
├── requirements.txt         # Project Python dependencies
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone [https://github.com/Asifarrrr/Risk_Project.git](https://github.com/Asifarrrr/Risk_Project.git)
cd Risk_Project
```

### 2. Set Up a Virtual Environment & Install Dependencies
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the Backend & Frontend (Unified Server)
Start the Uvicorn server from the root directory to launch the app 🚀:
```bash
uvicorn app.main:app --reload
```
*(Note: The server will automatically trigger your browser to open the dashboard page on startup!)*

### 4. Access the Application
* **Interactive Dashboard:** Open your browser and navigate to **`http://127.0.0.1:8000`** 🌐
* **API Documentation (Swagger UI):** Visit **`http://127.0.0.1:8000/docs`** 📄

---

## 🛠️ Tech Stack & Workflow

- **Python** – Core programming language 🐍.
- **FastAPI & Uvicorn** – Backend web server and REST API framework ⚡.
- **Pandas, NumPy & Scikit-learn** – Data preprocessing, feature encoding, and Random Forest classification 📈.
- **Joblib** – Model serialization and persistence (`.pkl` files) 💾.
- **HTML5, CSS3, JavaScript (Fetch API) & Chart.js** – Modern dark-themed interactive web dashboard 🖥️.

---

## 📌 Project Objectives

- Master the end-to-end Machine Learning pipeline from data cleaning and encoding to production deployment 🛠️.
- Transition from ad-hoc notebooks to structured, production-ready full-stack architecture 🏗️.
- Learn how to serve modern frontend interfaces directly through a FastAPI Python backend 🔗.
- Maintain clean version control workflows using Git and GitHub 🐙.

---

**This project was developed as part of my journey in Data Science and Machine Learning, combining predictive modeling, API development, and web UI integration into one practical portfolio piece 🌟.**
