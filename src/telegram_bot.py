import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
import numpy as np
import pandas as pd
import torch
import os
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

# Обработчик команды /start
async def start(update: Update, context):
    await update.message.reply_text(
"""New patient? Here to complain about your life?
Diagnosis: acute boredom, complicated by excessive free time."""
        )

# Обработчик текстовых сообщений
async def handle_message(update: Update, context):
    user_message = update.message.text
    bot_response = find_best_response(user_message)
    await update.message.reply_text(bot_response)

# Настройка логгирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Запуск бота
if __name__ == "__main__":
    # Укажите ваш токен здесь
    TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

    # Создаем приложение
    application = ApplicationBuilder().token(TOKEN).build()

    # Регистрируем обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Запускаем бота
    application.run_polling()