# VPg04. RAG-агент с Pinecone и внешним API

Учебный проект на Python 3.13: загрузка HTML-страницы в векторную базу, поиск по знаниям и агент LangChain с реальным API-инструментом.

## Основа проекта

- Из [VPg02-semantic-search](https://github.com/vtaranov/zerocoder-vpg02-semantic-search) адаптированы подключение Pinecone, проверка размерности 1536 и метрики cosine, получение эмбеддингов, upsert и query.
- Из VPf07-ai-agent использован подход «модель выбирает инструмент → Python вызывает API → результат возвращается модели», формат результатов `ok/data/error` и явная обработка ошибок. Циклом теперь управляет LangChain `create_agent`.
- Новый код: загрузка HTML, извлечение текста, чанкинг, метаданные, контроль повторной загрузки и инструменты `@tool`.

## Установка и настройка

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
cp .env.example .env
```

В `.env` заполните `OPENAI_API_KEY`, `PINECONE_API_KEY` и `PINECONE_INDEX`. Нужен существующий готовый индекс с dimension=1536 и metric=cosine. Можно использовать свой совместимый индекс, указав отдельный `PINECONE_NAMESPACE=vpg04-rag-demo`. Индекс автоматически не создаётся. `OPENAI_BASE_URL` задаёт совместимый адрес API; `CHAT_MODEL` должен быть доступен этому подключению. В проверке использовался `gpt-6-luna` из предыдущего проекта.

## Команды

```bash
.venv/bin/python app.py probe
.venv/bin/python app.py add-url https://www.python.org/doc/essays/blurb/
.venv/bin/python app.py search 'What is Python?'
.venv/bin/python app.py ask 'По базе знаний: какие особенности Python перечислены в статье? Укажи источник.'
.venv/bin/python app.py cat-fact
.venv/bin/python app.py ask 'Получи случайный факт о кошках через API и укажи источник.'
```

`RAGAgent` можно импортировать из `rag_agent.py`: методы `probe()`, `add_url(url)`, `search(query)` и `answer(question)` отделены от интерфейса. `answer` возвращает ответ и фактические результаты инструментов. Каждый вызов - отдельный вопрос, история диалога не сохраняется. Telegram-интерфейс и автоматическое выделение фактов о пользователе не входят в три обязательных пункта этого ДЗ.

## Как устроена загрузка

HTTP GET → BeautifulSoup → основной текст → чанки по 1000 символов с перекрытием 150 → OpenAI Embeddings `text-embedding-3-small` → Pinecone. В метаданных остаются URL, заголовок, текст, номер фрагмента и версия страницы.

ID зависит от URL, содержимого и номера фрагмента. Повторная загрузка той же страницы перезаписывает те же ID. При изменении страницы предыдущая версия удаляется после успешной записи новой. Для учебной последовательной работы; конкурентное обновление одного URL не предусмотрено.

Поиск возвращает три фрагмента. Агент должен опираться на их содержание и признавать отсутствие ответа. Фиксированный порог сходства не применяется: значение cosine не является вероятностью правильного ответа. Проверки на одном документе не доказывают качество на произвольной базе.

Загрузчик работает с публичным статическим HTML, без авторизации, OCR и выполнения JavaScript. Лимиты: 2 MB HTML, 100 000 символов текста, до пяти перенаправлений. Локальные адреса и нестандартные порты отклоняются, адрес проверяется при каждом перенаправлении. Это локальный учебный инструмент, не сервис для приёма недоверенных URL: отдельная защита от DNS rebinding не реализована.

## Проверки

```bash
.venv/bin/python -m unittest -v
.venv/bin/python -m pip check
.venv/bin/python verify_live.py
```

`verify_live.py` выполняет реальные запросы к OpenAI, Pinecone, Python.org и catfact.ninja; расходует квоту API и записывает учебные данные в namespace VPg04. После запуска полный протокол создаётся локально в `evidence/live-results.json`. Результаты исходной проверки кратко описаны в [VALIDATION.md](VALIDATION.md). Полные журналы и учебный отчёт в репозитории не публикуются. `requirements-lock.txt` фиксирует проверенные версии.

В репозитории находятся код, инструкция, пример настроек и краткие итоги проверки. Ключи, `.venv` и локальные протоколы исключены через `.gitignore`. Трассировка LangSmith отключена, вызовы модели используют `store=False`.

## Документация

- [LangChain: агенты](https://docs.langchain.com/oss/python/langchain/agents)
- [LangChain: инструменты и @tool](https://docs.langchain.com/oss/python/langchain/tools)
- [Pinecone: upsert и query](https://sdk.pinecone.io/python/how-to/vectors/upsert-and-query.html)
- [OpenAI: Responses API](https://developers.openai.com/api/docs/guides/migrate-to-responses#additional-differences)
