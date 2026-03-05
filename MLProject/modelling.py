import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# Fungsi helper untuk men-download dan pre-process data sementara di dalam MLProject
def get_data(data_file):
    print(f"Loading data from {data_file}...")
    train_df = pd.read_csv(data_file)
    X = train_df.drop('target', axis=1)
    y = train_df['target']
    return X, y

def main(data_file):
    import dagshub
    import sys
    import contextlib
    with contextlib.redirect_stdout(sys.stderr):
        dagshub.init(repo_owner='antsig', repo_name='Model-SML', mlflow=True)
        
    X, y = get_data(data_file)

    # Clear MLFLOW_RUN_ID environment variable so it doesn't conflict with DagsHub
    if "MLFLOW_RUN_ID" in os.environ:
        del os.environ["MLFLOW_RUN_ID"]

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
    import sys
    # MLflow on Windows might split the path at spaces or pass literal quotes
    # Rejoin array to handle space splitting and strip literal quotes
    data_file = " ".join(sys.argv[1:]).strip('"').strip("'") if len(sys.argv) > 1 else None
    if not data_file:
        raise ValueError("Missing data_file parameter")
    main(data_file)
