import os
import dagshub
import mlflow
from mlflow.models import build_docker

def main():
    # Inisialisasi DagsHub untuk memastikan MLflow memiliki otentikasi yang benar
    # untuk mengunduh artefak (model) dari server DagsHub.
    dagshub.init(repo_owner='antsig', repo_name='Model-SML', mlflow=True)
    
    run_id = os.environ.get("RUN_ID")
    if not run_id:
        print("Error: RUN_ID environment variable is missing")
        return
        
    model_uri = f"runs:/{run_id}/model"
    print(f"Building Docker image for model URI: {model_uri}...")
    
    # Gunakan Python API dari MLflow untuk mem-build Docker,
    # karena ini berjalan di dalam memori/proses di mana patch DagsHub aktif.
    build_docker(model_uri=model_uri, name="sml-model:latest")
    
if __name__ == "__main__":
    main()
