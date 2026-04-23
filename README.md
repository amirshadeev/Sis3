# SIS-3 — ML System: Streamlit + MLflow
### Расширение Practical Task 6

## Структура проекта

```
sis3/
├── app/
│   └── app.py                    ← PT6 app.py (без изменений)
├── frontend/
│   └── streamlit_app.py          ← NEW: Streamlit UI
├── training/
│   └── train_with_mlflow.py      ← NEW: PT6 train + MLflow logging
├── mlflow_data/                  ← создаётся автоматически
├── docker-compose.yml            ← 4 сервиса
├── Dockerfile.trainer
├── Dockerfile.api                ← PT6 Dockerfile (адаптирован)
├── Dockerfile.frontend
├── requirements.trainer.txt
├── requirements.api.txt          ← PT6 requirements.txt
└── requirements.frontend.txt
```

## Запуск

```bash
docker-compose up --build
```

| Сервис | URL |
|--------|-----|
| Streamlit UI | http://localhost:8501 |
| FastAPI Swagger | http://localhost:8000/docs |
| MLflow Dashboard | http://localhost:5000 |

## Порядок запуска

```
mlflow ──healthcheck──► trainer ──► api ──healthcheck──► frontend
```

1. **mlflow** — Tracking Server + SQLite + artifact store
2. **trainer** — обучает модель, логирует в MLflow, сохраняет model.joblib
3. **api** — PT6 app.py загружает model.joblib (без изменений)
4. **frontend** — Streamlit UI на :8501

## MLflow — что логируется

**Experiment:** `iris_randomforest`

**Params:** n_estimators=100, random_state=42, test_size=0.2

**Metrics:** accuracy, f1_macro, precision, recall

**Artifacts:** model.joblib, classification_report.txt, MLflow sklearn model

**Model Registry:** `IrisClassifier` (версия автоинкрементируется)

## PT6 API (без изменений)

```
GET  /            → health check
GET  /model-info  → metadata
POST /predict     → предсказание

POST body: {"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}
Response:  {"predicted_class_id":0,"predicted_class_name":"setosa","probabilities":{...}}
```

## Остановка

```bash
docker-compose down       # остановить
docker-compose down -v    # + удалить volumes
```
