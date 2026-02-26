import dagshub
import mlflow

def main():
    import sys
    import contextlib

    try:
        # Patch MLflow with DagsHub credentials to allow searching artifacts remotely
        # Redirect stdout to stderr to avoid polluting the RUN_ID capture
        with contextlib.redirect_stdout(sys.stderr):
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
        if r.data.tags.get("mlflow.runName") == "MLProject_Run":
            print(r.info.run_id)
            return
            
    print("")

if __name__ == "__main__":
    main()
