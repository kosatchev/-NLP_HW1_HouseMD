import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel


# Константы
DATA_PATH = "../data/processed/context_answer.csv"
OUTPUT_DIR = "../models"
BATCH_SIZE = 16
MODEL_NAME = 'bert-base-uncased'


# Загрузка предобученной модели и токенизатора
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME)

# Переносим модель на GPU, если доступно
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


def get_bert_embedding(texts, model, tokenizer, batch_size=16):
    """Получает эмбеддинги для нескольких текстов с помощью BERT."""
    # Токенизация в батчах
    inputs = tokenizer(
        texts, 
        return_tensors='pt', 
        truncation=True, 
        padding=True, 
        max_length=512
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Переносим данные на GPU

    with torch.no_grad():
        outputs = model(**inputs)

    # Используем среднее значение эмбеддингов токенов как вектор текста
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Переносим обратно на CPU
    return embeddings


def load_data(data_path):
    """Загружает обработанные данные."""
    return pd.read_csv(data_path)


def save_embeddings(embeddings, output_path):
    """Сохраняет эмбеддинги в файл."""
    if embeddings.ndim > 2:
        raise ValueError(f"Ожидался двумерный массив, но получен массив формы {embeddings.shape}")
    np.save(output_path, embeddings)


def load_embeddings(input_path):
    """Загружает эмбеддинги из файла."""
    return np.load(input_path)


def train(data_path, output_dir, batch_size=16):
    """Обучает модель и сохраняет эмбеддинги."""
    # Загружаем данные
    df = load_data(data_path)

    # Получаем эмбеддинги для всех ответов с использованием tqdm и батчей
    response_embeddings = []
    for i in tqdm(range(0, len(df), batch_size), desc="Создание эмбеддингов"):
        batch_responses = df['response'][i:i + batch_size].tolist()
        embeddings = get_bert_embedding(batch_responses, model, tokenizer, batch_size)
        response_embeddings.append(embeddings)

    response_embeddings = np.vstack(response_embeddings)  # Объединяем батчи в один массив

    # Проверяем форму массива
    print(f"Форма массива эмбеддингов: {response_embeddings.shape}")

    # Сохраняем эмбеддинги
    os.makedirs(output_dir, exist_ok=True)
    save_embeddings(response_embeddings, os.path.join(output_dir, 'response_embeddings.npy'))
    print(f"Эмбеддинги сохранены в {output_dir}")


def infer(context, response_embeddings, df, model, tokenizer):
    """Находит наиболее подходящий ответ для заданного контекста."""
    # Получаем эмбеддинг контекста
    context_embedding = get_bert_embedding([context], model, tokenizer)
    
    # Вычисляем косинусное сходство между контекстом и всеми ответами
    similarities = cosine_similarity(context_embedding, response_embeddings)[0]
    
    # Находим индекс наиболее подходящего ответа
    best_index = np.argmax(similarities)
    return df.iloc[best_index]['response']


if __name__ == "__main__":
    # Обучаем модель и сохраняем эмбеддинги
    train(DATA_PATH, OUTPUT_DIR, BATCH_SIZE)

    # Пример инференса
    response_embeddings = load_embeddings(os.path.join(OUTPUT_DIR, 'response_embeddings.npy'))
    df = load_data(DATA_PATH)
    context = "Why are you late?"
    best_response = infer(context, response_embeddings, df, model, tokenizer)

    print(f"Контекст: {context}")
    print(f"Ответ: {best_response}")