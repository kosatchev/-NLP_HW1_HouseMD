# Отчет о выполнении задания

## Описание проекта

Этот проект представляет собой retrieval-based чат-бота, который имитирует стиль и манеру речи доктора Хауса из сериала "Доктор Хаус". Бот использует BERT для получения эмбеддингов текста и косинусное сходство для поиска наиболее подходящих ответов. Проект включает:

- Веб-интерфейс на FastAPI.
- Telegram-бот.
- Бот в виде командной строки.
- Docker-контейнеры для развертывания на сервере.


## Данные

- **Источник**: Транскрипции сериала "Доктор Хаус" с Kaggle.
- **Объем**: 8 сезонов, ~10 000 реплик.
- **Предобработка**:
  - Очистка текста от лишних символов и тегов.
  - Создание пар "контекст-ответ".


## Модель

**Архитектура**: BERT для получения эмбеддингов, косинусное сходство для поиска ответов.


## Веб-приложение

![Веб-приложение](images/web_app.png)

**Технологии**: FastAPI, HTML, JavaScript.

### Функционал

  - Отправка сообщений по нажатию **Enter** или кнопки **Send**.
  - Отображение истории сообщений.
  
### Запуск

```bash
python src/web_app.py
```


## Telegram-бот

![Бот](images/telegram_bot.png)

**Технологии**: python-telegram-bot.

### Функционал

- Ответы на сообщения пользователя.
- Поддержка команды /start.

### Запуск

```bash
python src/telegram_bot.py
```


## Инференс командной строки

![Командная строка](images/cli_inference.png)

### Функционал

- Общение с моделью через терминал.
- Поддержка команды exit для завершения.

Запуск:
```bash
python src/cli_inference.py
```

## Docker

### Контейнеры:

- house-md-telegram-bot: Telegram-бот.
- house-md-web-app: Веб-приложение.

**Управление**: Docker Compose.

### Запуск

```bash
docker-compose up --build
```


## Зависимости

Зависимости разделены для разных частей проекта:

- Telegram-бот: requirements_bot.txt
- Веб-приложение: requirements_web.txt
- Обучение и обработка модели: requirements_train.txt

Для установки зависимостей локально:

```bash
# Для Telegram-бота
pip install -r requirements_bot.txt

# Для веб-приложения
pip install -r requirements_web.txt

# Для обучения и обработки модели
pip install -r requirements_train.txt
```

## Графики
В результате тестирования, получилось такое распределение
![График](images/similarities.png)


## Выводы
- Бот успешно имитирует стиль доктора Хауса.
- Веб-приложение и Telegram-бот работают стабильно.
- Docker-контейнеры позволяют развернуть приложения на любом сервере.


## Автор

Косачев Дмитрий Викторович


### **Итоговая структура проекта**
```
house-md-chatbot/
├── processed/
│ ├── raw/
│ │ └── context_answer.csv
│ └── data/
│   ├── season1.csv
│   ├── ...
│   └── season8.csv
├── images/
│ ├── cli_inference.png
│ ├── similarities.png
│ ├── telegram_bot.png
│ └── web_app.png
├── models/
│ └── response_embeddings.npy
├── notebooks/
│ ├── data_preprocessing.ipynb
│ └── model_training.ipynb
├── src/
│ ├── web_app.py
│ ├── telegram_bot.py
│ ├── preprocess.py
│ ├── train.py
│ ├── validate.py
│ └── cli_inference.py
├── templates/
│ └── index.html
├── .env
├── .gitignore
├── docker-compose.yml
├── Dockerfile.bot
├── Dockerfile.web
├── requirements_bot.txt
├── requirements_web.txt
└── README.md
```