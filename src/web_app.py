from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# Загрузка модели и данных
MODEL_NAME = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Загрузка эмбеддингов и данных
response_embeddings = np.load("./models/response_embeddings.npy")
df = pd.read_csv("./data/processed/context_answer.csv")

# Определяем модель запроса
class ChatRequest(BaseModel):
    message: str

# Создаем FastAPI приложение
app = FastAPI()

# Монтируем статические файлы (если нужно)
# app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Возвращает HTML-страницу с интерфейсом чата."""
    with open("templates/index.html", "r") as file:
        return HTMLResponse(content=file.read())

@app.post("/chat")
async def chat(request: ChatRequest):
    """Возвращает ответ на сообщение пользователя."""
    context = request.message
    if not context:
        raise HTTPException(status_code=400, detail="Сообщение не может быть пустым")

    # Получаем эмбеддинг контекста
    context_embedding = get_bert_embedding([context], model, tokenizer)
    
    # Вычисляем косинусное сходство
    similarities = cosine_similarity(context_embedding, response_embeddings)[0]
    best_index = np.argmax(similarities)
    best_response = df.iloc[best_index]['response']

    return {"response": best_response}

def get_bert_embedding(texts, model, tokenizer):
    """Получает эмбеддинги для текстов с помощью BERT."""
    inputs = tokenizer(
        texts, 
        return_tensors='pt', 
        truncation=True, 
        padding=True, 
        max_length=512
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)