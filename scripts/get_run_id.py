import dagshub
import mlflow

def main():
    try:
        # Patch MLflow with DagsHub credentials to allow searching artifacts remotely
        dagshub.init(repo_owner='antsig', repo_name='Model-SML', mlflow=True)
    except Exception as e:
        pass # Ignore failure if auth is already handled or fails locally
        
    c = mlflow.tracking.MlflowClient()
    exps = [e.experiment_id for e in c.search_experiments()]
    if not exps:
        print("")
        return
        
    runs = c.search_runs(exps, order_by=["start_time desc"])
    for r in runs:
        try:
            artifacts = [a.path for a in c.list_artifacts(r.info.run_id, "")]
            if "model" in artifacts:
                print(r.info.run_id)
                return
        except Exception:
            # If we lack permission to list artifacts for an older run, skip
            continue
            
    print("")

if __name__ == "__main__":
    main()
