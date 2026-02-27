import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import urllib.request

# Fungsi helper untuk men-download dan pre-process data sementara di dalam MLProject
def get_data():
    train_df = pd.read_csv('../Membangun_model/iris_preprocessing/train.csv')
    X = train_df.drop('species', axis=1)
    y = train_df['species']
    return X, y

def main():
    import dagshub
    import sys
    import contextlib
    with contextlib.redirect_stdout(sys.stderr):
        dagshub.init(repo_owner='antsig', repo_name='Model-SML', mlflow=True)
        
    X, y = get_data()

    with mlflow.start_run() as run:
        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)
            
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X, y)
        preds = clf.predict(X)
        acc = accuracy_score(y, preds)
        print(f"Accuracy on train set: {acc}")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_param("random_state", 42)
        mlflow.sklearn.log_model(clf, "model")
        
        # Explicitly save model locally to guarantee Docker build bypassing DagsHub download issues
        import shutil
        if os.path.exists("local_model"):
            shutil.rmtree("local_model")
        mlflow.sklearn.save_model(clf, "local_model")

if __name__ == "__main__":
    main()
