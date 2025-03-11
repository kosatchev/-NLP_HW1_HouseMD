import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import os

def validate(response_embeddings, df, model, tokenizer, test_size=0.2, batch_size=16):
    """Оценивает качество модели на тестовой выборке."""
    
    # Определяем устройство (GPU, если доступно, иначе CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Перемещаем модель на устройство

    # Разделяем данные на обучающую и тестовую выборки
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    # Получаем эмбеддинги для тестовых данных с использованием батчей
    test_embeddings = []
    for i in tqdm(range(0, len(test_df), batch_size), desc="Создание тестовых эмбеддингов"):
        batch_contexts = test_df['context'][i:i + batch_size].tolist()
        
        # Токенизация и перемещение на устройство
        inputs = tokenizer(batch_contexts, return_tensors='pt', padding=True, truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}  # Перемещаем входные данные на устройство
        
        # Получаем эмбеддинги
        with torch.no_grad():
            batch_embeddings = model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()  # Перемещаем результаты обратно на CPU для дальнейшей обработки
        
        test_embeddings.append(batch_embeddings)
    test_embeddings = np.vstack(test_embeddings)

    # Вычисляем косинусное сходство для тестовых данных
    similarities = cosine_similarity(test_embeddings, response_embeddings)
    best_indices = np.argmax(similarities, axis=1)

    # Оцениваем точность
    correct = 0
    for i, index in enumerate(best_indices):
        if index < len(train_df) and train_df.iloc[index]['response'] == test_df.iloc[i]['response']:
            correct += 1
    accuracy = correct / len(test_df)
    print(f"Точность модели на тестовой выборке: {accuracy:.2f}")

    # Визуализация распределения косинусного сходства
    plt.figure(figsize=(10, 6))
    plt.hist(similarities.flatten(), bins=50, alpha=0.7, label="Косинусное сходство")
    plt.xlabel("Косинусное сходство")
    plt.ylabel("Частота")
    plt.title("Распределение косинусного сходства")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'validation_histogram.png'))  # Сохраняем график
    plt.show()

    return accuracy

if __name__ == "__main__":

    # Загружаем эмбеддинги и данные
    response_embeddings = load_embeddings(os.path.join(OUTPUT_DIR, 'response_embeddings.npy'))
    df = load_data(DATA_PATH)

    # Пример инференса
    context = "Why are you late?"
    best_response = infer(context, response_embeddings, df, model, tokenizer)
    print(f"Контекст: {context}")
    print(f"Ответ: {best_response}")

    # Валидация модели
    validate(response_embeddings, df, model, tokenizer)