import pandas as pd
import mlflow
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import argparse
import json



def evaluate_model(test_data_path, run_id):
    """Оценка модели и логирование метрик"""
    test_data = pd.read_csv(test_data_path)
    X_test = test_data.drop('Survived', axis=1)
    y_test = test_data['Survived']
    
    with mlflow.start_run(run_id=run_id):
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/titanic_model")
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred)
        }
        
        mlflow.log_metrics(metrics)
        print(f"Метрики модели: {metrics}")

        with open("metrics.json", "w") as f:
            json.dump(metrics, f)

def get_latest_run_id():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    runs = mlflow.search_runs(max_results=1)
    return runs.iloc[0].run_id if not runs.empty else None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data", type=str, required=True)
    parser.add_argument("--run-id", type=str, required=True)
    args = parser.parse_args()

    run_id = args.run_id if args.run_id else get_latest_run_id()
    if not run_id:
        raise ValueError("No MLflow runs found and no run-id provided")
    
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    evaluate_model(args.test_data, args.run_id)