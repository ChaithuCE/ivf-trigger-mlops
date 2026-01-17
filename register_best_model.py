import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "IVF_Trigger_Prediction"
MODEL_NAME = "ivf_trigger_model"  # name for registry

def main():
    client = MlflowClient()

    # 1) find best run again (by roc_auc)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.roc_auc DESC"],
        max_results=1,
    )
    best_run = runs[0]
    best_run_id = best_run.info.run_id
    best_roc_auc = best_run.data.metrics["roc_auc"]

    print("Best run_id:", best_run_id)
    print("Best roc_auc:", best_roc_auc)

    # 2) register this run as a model in registry
    model_uri = f"runs:/{best_run_id}/model"
    result = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)

    print("Registered model name:", result.name)
    print("Registered model version:", result.version)

if __name__ == "__main__":
    main()
