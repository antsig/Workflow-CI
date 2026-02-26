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
    
    # Debug: List out the real artifacts inside the run
    c = mlflow.tracking.MlflowClient()
    try:
        real_artifacts = c.list_artifacts(run_id)
        print("Real artifacts found in run:")
        for a in real_artifacts:
            print(f" - {a.path} (is_dir: {a.is_dir})")
    except Exception as e:
        print(f"Failed to list artifacts: {e}")

    try:
        local_path = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/model", dst_path="./model_workspace")
        print(f"Model downloaded successfully to {local_path}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    main()
