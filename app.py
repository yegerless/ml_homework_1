import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from fastapi import FastAPI, File, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, RootModel
from typing import List
import io


# Загрузка обученной модели
MODEL = joblib.load('model.pickle')

app = FastAPI()


# Удалил из базовой модели признак selling_price так как его наличие во входных данных кажется странным, 
# ведь сервис должен его прогнозировать, это целевая переменная
class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float



class Items(RootModel):
    root: List[Item]



def pydantic_model_to_df(item: Item) -> pd.DataFrame:
    'Принимает объект класса Item и преобразует его в pandas DataFrame'
    return pd.DataFrame([jsonable_encoder(item)])


def get_clean_data(data: pd.DataFrame) -> pd.DataFrame:
    'Принимает pandas DataFrame и приводит его к виду, с которым может работать обученная заранее модель'
    data.drop('torque', axis='columns', inplace=True)
    data['mileage'] = data['mileage'].apply(lambda x: float(str(x).split()[0]))
    data['engine'] = data['engine'].apply(lambda x: float(str(x).split()[0]))
    data['max_power'] = data['max_power'].apply(lambda x: float(str(x).split()[0]) if x != ' bhp' else np.nan)
    return data


def get_prediction(data: pd.DataFrame) -> np.ndarray:
    'Принимает подготовленный pandas DataFrame с одним или более объектом и возвращает numpy ndarray с предсказаниями selling_price'
    prediction = MODEL.predict(data)
    return prediction



@app.post(path="/predict_item")
def predict_item(item: Item) -> float:
    item = pydantic_model_to_df(item)
    item = get_clean_data(item)
    prediction = float(get_prediction(item)[0])
    return prediction


@app.post("/predict_items", response_class=StreamingResponse)
def predict_items(file: UploadFile = File(...)):
    data = Items.model_validate(pd.read_csv(file.file).to_dict(orient='records'))
    data = pd.DataFrame(data.model_dump())
    data = get_clean_data(data)
    data['prediction'] = get_prediction(data)
    stream = io.StringIO()
    data.to_csv(stream, index=False)
    headers = {"Content-Disposition": "attachment; filename=file.csv"}
    response = StreamingResponse(iter([stream.getvalue()]), headers=headers, media_type='txt/csv')
    return response