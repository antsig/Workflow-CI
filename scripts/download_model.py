import os
import dagshub
import mlflow

def main():
    # Initialize DagsHub to patch MLflow artifact downloading auth
    dagshub.init(repo_owner='antsig', repo_name='Model-SML', mlflow=True)
    
    run_id = os.environ.get("RUN_ID")
    if not run_id:
        print("Error: RUN_ID environment variable is missing")
        return
        
    print(f"Downloading model artifact for run {run_id}...")
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/model", dst_path=".")
    print(f"Model downloaded successfully to {local_path}")

if __name__ == "__main__":
    main()
