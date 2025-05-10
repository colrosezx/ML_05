import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow
import yaml
import argparse
from mlflow.models.signature import infer_signature

def train_model(data_path, model_dir, params):
    """Обучение модели с логированием в MLflow"""
    df = pd.read_csv(data_path)
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with mlflow.start_run():
        mlflow.log_params(params)

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=model_dir,
            signature=signature,
            input_example=X_train.iloc[:5]
        )
        
        with open("run_id.txt", "w") as f:
            f.write(mlflow.active_run().info.run_id)
        
        return X_test, y_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model-dir", type=str, default="titanic_model")
    parser.add_argument("--params-file", type=str, default="params.yaml")
    args = parser.parse_args()
    
    with open(args.params_file) as f:
        params = yaml.safe_load(f)["train"]
    
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Titanic_Survival_Prediction")
    
    X_test, y_test = train_model(args.data, args.model_dir, params)

    pd.concat([X_test, y_test], axis=1).to_csv("test_data.csv", index=False)