# MLFLOW Models

```bash
# serve the model via REST
mlflow models serve -m "path" --port 8000 --env-manager=local
mlflow models serve -m "file:///C:/Users/BESTWAY/Downloads/Technical Project/ML Flow/Sentiment_Analysis_Artifacts/e3979f1e765d42e7997ca81dc4e7af7d/artifacts/Random Forest" --port 8000 --env-manager=local

# it will open in this link
http://localhost:8000/invocations
```
