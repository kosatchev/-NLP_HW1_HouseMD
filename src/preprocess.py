import os
import re
import pandas as pd


def load_data(raw_data_dir):
    """Загружает все CSV-файлы из директории и объединяет их в один DataFrame."""

    all_files = [os.path.join(raw_data_dir, f) for f in os.listdir(raw_data_dir) if f.endswith('.csv')]
    df_list = [pd.read_csv(file, encoding='ISO-8859-1') for file in all_files]
    return pd.concat(df_list, ignore_index=True)


def clean_text(df):
    """Очищает текст от лишних символов и тегов."""

    # Удаляем строки с NaN в столбце 'line'
    full_clean_text = df.dropna(subset=['name', 'line']).reset_index(drop=True)

    # Применяем очистку к столбцу 'line' и удаляем лишние пробелы и символы
    full_clean_text.loc[:, 'line'] = full_clean_text['line'].apply(lambda x: ' '.join(re.sub(r'\[.*?\]|\(.*?\)', '', x).split()))

    return full_clean_text


def create_context_response_pairs(df, character="House"):
    """Создает пары 'контекст-ответ' для выбранного персонажа."""

    pairs = []
    for i in range(1, len(df)):
        if df.loc[i, 'name'] == character:
            context = df.loc[i - 1, 'line']
            response = df.loc[i, 'line']
            if context and response:  # Пропускаем пустые строки
                pairs.append({'context': context, 'response': response})
    return pd.DataFrame(pairs)


def save_data(df, output_path):
    """Сохраняет DataFrame в CSV-файл."""

    df.to_csv(output_path, index=False)


def preprocess_data(raw_data_dir, output_path, character="House"):
    """Основная функция для предобработки данных."""

    # Загружаем данные
    full_text = load_data(raw_data_dir)

    # Очищаем текст
    full_clean_text = clean_text(full_text)

    # Создаем пары "контекст-ответ"
    pairs_df = create_context_response_pairs(full_clean_text, character)

    # Сохраняем обработанные данные
    os.makedirs(output_path, exist_ok=True)
    
    save_data(pairs_df, os.path.join(output_path, 'context_answer.csv'))

if __name__ == "__main__":
    raw_data_dir = "./data/raw/"
    output_path = "./data/processed/"
    character = 'House'
    preprocess_data(raw_data_dir, output_path, character=character)