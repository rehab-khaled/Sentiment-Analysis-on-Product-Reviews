import mlflow

if __name__ == '__main__':
    mlflow.create_experiment(
        name = 'Sentiment_Analysis_Experiment',
        artifact_location = 'Sentiment_Analysis_Artifacts',
        tags = {'env':'dev', 'version':'1.0.0'}
        )
