from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

from custom_preprocessor import SelectiveNumericTransformer

app = FastAPI(title='Heart Disease Predictor')

pipeline = joblib.load('heart_disease_pipeline.pkl')

from fastapi import Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory='templates')

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse('index.html', {'request': request}) # This renders templates/index.html

@app.post('/predict', response_class=HTMLResponse)
def predict(
    request: Request,
    age: int = Form(...),
    sex: int = Form(...),
    cp: int = Form(...),
    trestbps: float = Form(...),
    chol: float = Form(...),
    fbs: int = Form(...),
    restecg: int = Form(...),
    thalach: float = Form(...),
    exang: int = Form(...),
    oldpeak: float = Form(...)
):
    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak
    }])

    prob = pipeline.predict_proba(input_df)[0,1]
    prediction = round(prob * 100, 2)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": prediction,
            "risk": "High Risk" if prob >= 0.5 else "Low Risk"
        }
    )


#Open PowerShell in Admin Mode > Enter "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser"> Give 'Y', it'll allow virtual envirnoment to run in terminals


# installing uvicorn : python -m pip install fastapi uvicorn
# In terminal run : uvicorn <filename>:<app_variabel name> --reload
# in our case : python -m uvicorn App:app --reload