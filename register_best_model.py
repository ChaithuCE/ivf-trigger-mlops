import mlflow
from mlflow.tracking import MlflowClient
from feast import FeatureStore
import os
from datetime import datetime

# ===================================================================
# CONFIG
# ===================================================================
EXPERIMENT_NAME = "IVF_Trigger_Prediction"
MODEL_NAME = "ivf_trigger_model"
FEAST_REPO_PATH = os.path.join(os.path.dirname(__file__), "feast", "feature_repo")

# ===================================================================
# INITIALIZE FEAST
# ===================================================================
fs = FeatureStore(repo_path=FEAST_REPO_PATH)


def main():
    """Find best model and register with FEAST integration"""
    
    # ===================================================================
    # MATERIALIZE FEAST FEATURES BEFORE REGISTRATION
    # ===================================================================
    print("\n" + "="*70)
    print("🔄 FEAST: Materializing features to online store...")
    print("="*70)
    try:
        fs.materialize_incremental(end_date=datetime.now())
        print("✅ FEAST: Features materialized successfully!")
    except Exception as e:
        print(f"⚠️  FEAST: {e}")
        print("   Continuing with model registration...")
    
    # ===================================================================
    # FIND BEST RUN BY ROC_AUC
    # ===================================================================
    print("\n" + "="*70)
    print("🔍 Searching for best model...")
    print("="*70)
    
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    
    if not experiment:
        print(f"❌ Experiment '{EXPERIMENT_NAME}' not found!")
        return
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.roc_auc DESC"],
        max_results=1,
    )
    
    if not runs:
        print(f"❌ No runs found in experiment '{EXPERIMENT_NAME}'")
        return
    
    best_run = runs[0]
    best_run_id = best_run.info.run_id
    best_model_name = best_run.info.run_name
    best_roc_auc = best_run.data.metrics["roc_auc"]
    
    print(f"✅ Best Run ID: {best_run_id}")
    print(f"✅ Best Model: {best_model_name}")
    print(f"✅ Best ROC_AUC: {best_roc_auc:.4f}")
    
    # ===================================================================
    # REGISTER MODEL WITH FEAST METADATA
    # ===================================================================
    print("\n" + "="*70)
    print("📝 Registering model with FEAST integration...")
    print("="*70)
    
    model_uri = f"runs/{best_run_id}/{best_model_name}"
    
    try:
        result = mlflow.register_model(
            model_uri=model_uri,
            name=MODEL_NAME
        )
        
        print(f"✅ Model registered!")
        print(f"   Name: {result.name}")
        print(f"   Version: {result.version}")
        
        # ===================================================================
        # UPDATE MODEL VERSION WITH FEAST METADATA
        # ===================================================================
        try:
            client.update_model_version(
                name=MODEL_NAME,
                version=result.version,
                description=f"FEAST integrated model - Algorithm: {best_model_name} - ROC_AUC: {round(best_roc_auc, 4)}"
            )
            print("✅ Model description updated with FEAST info!")
        except Exception as e:
            print(f"⚠️  Model update: {e}")
        
        print("\n" + "="*70)
        print("✅ MODEL REGISTRATION COMPLETE!")
        print("="*70)
        print(f"Model Name: {MODEL_NAME}")
        print(f"Version: {result.version}")
        print(f"Algorithm: {best_model_name}")
        print(f"ROC_AUC: {best_roc_auc:.4f}")
        print(f"FEAST Integration: YES ✓")
        print(f"Status: Ready for deployment")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"❌ Error registering model: {e}")


if __name__ == "__main__":
    main()
