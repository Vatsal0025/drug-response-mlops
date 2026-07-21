from fastapi import FastAPI
from src.api.schema import DrugInput
from src.api.inference import predict_gene_expression
app = FastAPI()


@app.get("/")
def home():
    return {
        "message": "Drug Response Prediction API is running"
    }

@app.post("/predict")
def predict(data: DrugInput):

    return predict_gene_expression(data)