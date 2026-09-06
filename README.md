# 💼 AI Salary Prediction

An end-to-end Machine Learning application that predicts the estimated annual salary of AI/ML professionals based on their professional profile.

The project combines **Machine Learning, FastAPI, Pydantic, Docker, and Streamlit** to create a complete ML deployment pipeline.

---

## 🚀 Project Overview

Salary prediction is a regression problem where multiple professional and organizational factors can influence compensation.

This project takes information such as:

- Job title
- Experience level
- Years of experience
- Employment type
- Company location
- Employee residence
- Education level
- Company size
- Industry

and uses a trained Machine Learning regression model to estimate the expected annual salary in USD.

### Application Architecture

```text
                  ┌──────────────────────┐
                  │  Streamlit Frontend  │
                  │      Port 8501       │
                  └──────────┬───────────┘
                             │
                             │ HTTP POST
                             ▼
                  ┌──────────────────────┐
                  │     FastAPI API      │
                  │      Port 8000       │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  Pydantic Validation │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Input Transformation │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  Trained ML Model    │
                  │       .pkl           │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Predicted Salary USD │
                  └──────────────────────┘