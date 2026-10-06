# PrEng. RecSys

Группа 11-411. Игнатьев Арсений, Николаева Полина.

Рекомендации для интернет-магазина: главная страница и карточка товара. Своего обучения большой модели нет. Есть контентная полка, соседи по покупкам, гибрид и простые правила: популярное, сезон, категория, бренд, «с этим покупают», недавно смотрели.

Кто что делал: Игнатьев — content-based, user-based CF и полки «популярное», «тот же бренд», «недавно смотрели» (`content.py`, `collaborative.py`, часть `heuristics.py`). Николаева — item-based CF, гибрид, полки «сезон», «та же категория», «с этим покупают», тесты и метрики (`collaborative.py`, `hybrid.py`, `evaluate.py`, `tests/`). Проверяли запуск вместе.

Подробности, метрики и что получилось на примерах — в REPORT.md.

## Подходы

| Подход | Где | Что делает |
| --- | --- | --- |
| Content-based | `recsys/content.py` | похожие товары по тексту карточки |
| User-based CF | `recsys/collaborative.py` | соседи среди покупателей |
| Item-based CF | `recsys/collaborative.py` | соседи среди товаров |
| Hybrid | `recsys/hybrid.py` | смесь item-based, контента и популярного |
| Popular, сезон, категория, бренд, also bought, recently viewed | `recsys/heuristics.py` | полки без модели |

Данные лежат в `data/`: 91 товар и лог покупок. Модели читают товары и события. Скрытые вкусы из `users.csv` в обучение не входят, они нужны только для проверки в отчёте.

## Установка

Нужен Python 3.11 или новее. Видеокарта не нужна.

```text
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

В `requirements.txt` три пакета: `numpy`, `pandas`, `scikit-learn`.

## Запуск

Из корня проекта:

```text
.\.venv\Scripts\python.exe -m recsys.demo
```

Команда учит полки на прошлом каждого пользователя и печатает главную, карточку смартфона и блок «с этим покупают» с фильтром и без него. Попадания в отложенные покупки помечены в выводе.

По отдельности:

```text
.\.venv\Scripts\python.exe -m recsys.evaluate
.\.venv\Scripts\python.exe -m tests.test_recommenders
```

`evaluate` считает метрики и пишет `artifacts/summary.json` и `artifacts/metrics.csv`. Папка `artifacts/` в git не коммитится. Если CSV ещё нет, их собирает `python data/generate_data.py`.

## Файлы

```text
recsys/content.py
recsys/collaborative.py
recsys/hybrid.py
recsys/heuristics.py
recsys/service.py          главная и карточка товара
recsys/evaluate.py         метрики
recsys/demo.py
data/                      каталог и лог
REPORT.md                  разбор и числа
```
