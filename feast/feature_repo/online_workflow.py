from datetime import datetime, timezone
from feast import FeatureStore

def materialize_to_online():
    store = FeatureStore(repo_path=".")
    store.materialize_incremental(end_date=datetime.now(timezone.utc))
    print("Materialized features to online store.")

def fetch_online_example():
    store = FeatureStore(repo_path=".")
    example_patient_id = "P0041"

    feature_vector = store.get_online_features(
        features=[
            "ivf_trigger_features:Age",
            "ivf_trigger_features:AMH (ng/mL)",
            "ivf_trigger_features:Day",
            "ivf_trigger_features:Avg_Follicle_Size_mm",
            "ivf_trigger_features:Follicle_Count",
            "ivf_trigger_features:Estradiol_pg_mL",
            "ivf_trigger_features:Progesterone_ng_mL",
        ],
        entity_rows=[{"Patient_ID": example_patient_id}],
    ).to_dict()

    # Flatten for easier print
    flat = {k: v[0] for k, v in feature_vector.items()}
    print(f"Online features for {example_patient_id}:", flat)

if __name__ == "__main__":
    materialize_to_online()
    fetch_online_example()
