import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import urllib.request

# Fungsi helper untuk men-download dan pre-process data sementara di dalam MLProject
def get_data():
    os.makedirs('data', exist_ok=True)
    if not os.path.exists('data/iris.csv'):
        urllib.request.urlretrieve("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv", "data/iris.csv")
    df = pd.read_csv('data/iris.csv')
    X = df.drop('species', axis=1)
    y = df['species']
    return X, y

def main():
    X, y = get_data()

    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="MLProject_Run"):
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X, y)
        preds = clf.predict(X)
        acc = accuracy_score(y, preds)
        print(f"Accuracy on train set: {acc}")
        mlflow.sklearn.log_model(clf, "model")

if __name__ == "__main__":
    main()
