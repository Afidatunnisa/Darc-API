import io
from fastapi import FastAPI
import pandas as pd
import numpy as np
from pydantic import BaseModel
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import joblib
from keras.models import load_model
import json
from keras.layers import Dropout
from openpyxl import Workbook
import uvicorn

def predictGod(pengalaman, pendidikan, skillset):
    wb = Workbook()

# Select the active sheet
    sheet = wb.active
    treeData = [["Pengalaman Kerja", "Pendidikan", "Skillset"], [pengalaman, pendidikan, skillset]]
    for row in treeData:
        sheet.append(row)
        
    in_memory_file = io.BytesIO()
    wb.save(in_memory_file)

    # Seek to the beginning of the file before reading its content
    in_memory_file.seek(0)
    df_user_input = pd.read_excel(in_memory_file)

    model_loaded = load_model("nlp_model.h5")
    tokenizer = Tokenizer()
    with open("nlp_tokenizer.json", 'r') as f:
        tokenizer_config = json.load(f)
        tokenizer.word_index = tokenizer_config.get('word_index', {})

    label_encoder_job_title = joblib.load("nlp_label_encoder_job_title.pkl")
    label_encoder_education = joblib.load("nlp_label_encoder_education.pkl")
    scaler_experience = joblib.load("scaler_experience.pkl")

    df_user_input['Pendidikan'] = label_encoder_education.transform(df_user_input['Pendidikan'])
    df_user_input['Pengalaman Kerja'] = scaler_experience.transform(df_user_input[['Pengalaman Kerja']]).reshape(-1, 1)

    sequences_user_input = tokenizer.texts_to_sequences(df_user_input['Skillset'].fillna(''))
    maxlen_train = max(len(seq) for seq in sequences_user_input)
    X_text_user_input = pad_sequences(sequences_user_input, padding='post',maxlen=maxlen_train)

    X_other_user_input = df_user_input[['Pengalaman Kerja', 'Pendidikan']]
    X_user_input = pd.concat([X_other_user_input, pd.DataFrame(X_text_user_input)], axis=1)
    predictions = model_loaded.predict(X_user_input)

    # Decode predictions to job titles
    predicted_labels = label_encoder_job_title.inverse_transform(np.argmax(predictions, axis=1))

    # Add the predicted job titles to the user input DataFrame
    df_user_input['predicted_job_title'] = predicted_labels

    # Display the DataFrame with predicted job titles
    tania = df_user_input.loc[0, 'predicted_job_title']
    return tania

class Item(BaseModel):
    pengalaman_kerja: int
    pendidikan: str
    skillset: str


app = FastAPI()


@app.post("/predict/")
async def create_item(item: Item):
    return {
        "prediction": predictGod(item.pengalaman_kerja, item.pendidikan, item.skillset)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8080)
