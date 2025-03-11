import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import signal
import sys

# Загрузка модели и данных
MODEL_NAME = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Загрузка эмбеддингов и данных
response_embeddings = np.load("./models/response_embeddings.npy")
df = pd.read_csv("./data/processed/context_answer.csv")

# Функция для получения эмбеддингов
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

# Функция для поиска ответа
def find_best_response(context):
    """Находит наиболее подходящий ответ для заданного контекста."""
    context_embedding = get_bert_embedding([context], model, tokenizer)
    similarities = cosine_similarity(context_embedding, response_embeddings)[0]
    best_index = np.argmax(similarities)
    return df.iloc[best_index]['response']

# Обработчик сигнала SIGINT (Ctrl+C)
def signal_handler(sig, frame):
    print("\nЗавершение работы...")
    sys.exit(0)

# Основной цикл для взаимодействия с пользователем
def main():
    print(
"""New patient? Here to complain about your life?
Diagnosis: acute boredom, complicated by excessive free time.
Type 'exit' and go do something useful."""
        )
    while True:
        try:
            user_input = input("Вы: ")
            if user_input.lower() == "exit":
                print("Завершение работы...")
                break
            bot_response = find_best_response(user_input)
            print(f"Бот: {bot_response}")
        except KeyboardInterrupt:
            print("\nЗавершение работы...")
            break

if __name__ == "__main__":
    # Регистрируем обработчик сигнала SIGINT
    signal.signal(signal.SIGINT, signal_handler)
    main()