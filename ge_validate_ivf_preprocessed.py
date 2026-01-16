import os
import pandas as pd
import great_expectations as ge
import json

PROJECT_ROOT = r"C:\AI_IVF_Trigger_day"
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "ivf_trigger_preprocessed.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "quality", "ivf_trigger_ge_validation.json")


def main():
    df = pd.read_csv(CSV_PATH)

    gdf = ge.dataset.PandasDataset(df)

    # ---------- Table-level checks ----------
    gdf.expect_table_row_count_to_be_between(min_value=600, max_value=800)
    gdf.expect_table_column_count_to_be_between(min_value=15, max_value=20)

    # ---------- Core schema and non-null checks ----------
    numeric_cols = [
        "age",
        "amh_ng_ml",
        "day",
        "avg_follicle_size_mm",
        "follicle_count",
        "estradiol_pg_ml",
        "progesterone_ng_ml",
        "trigger_recommended",
    ]
    for col in numeric_cols:
        gdf.expect_column_to_exist(col)
        gdf.expect_column_values_to_not_be_null(col)
        gdf.expect_column_values_to_be_in_type_list(col, ["int64", "float64", "Int64"])

    gdf.expect_column_to_exist("patient_id")
    gdf.expect_column_values_to_not_be_null("patient_id")
    gdf.expect_column_values_to_be_of_type("patient_id", "object")

    # ---------- Clinical range checks ----------
    gdf.expect_column_values_to_be_between("age", min_value=18, max_value=50)
    gdf.expect_column_values_to_be_between("amh_ng_ml", min_value=0.1, max_value=15)
    gdf.expect_column_values_to_be_between("day", min_value=2, max_value=14)
    gdf.expect_column_values_to_be_between("avg_follicle_size_mm", min_value=8, max_value=30)
    gdf.expect_column_values_to_be_between("follicle_count", min_value=1, max_value=60)
    gdf.expect_column_values_to_be_between("estradiol_pg_ml", min_value=20, max_value=6000)
    gdf.expect_column_values_to_be_between("progesterone_ng_ml", min_value=0.1, max_value=5.0)

    # ---------- Target encoding ----------
    gdf.expect_column_values_to_be_in_set("trigger_recommended", value_set=[0, 1])

    # ---------- Engineered categorical features (if present) ----------
    if "age_group" in gdf.columns:
        gdf.expect_column_values_to_not_be_null("age_group")
        gdf.expect_column_distinct_values_to_be_in_set(
            "age_group", ["<30", "30-34", "35-37", "38-40", ">40"]
        )

    if "amh_group" in gdf.columns:
        gdf.expect_column_values_to_not_be_null("amh_group")
        gdf.expect_column_distinct_values_to_be_in_set(
            "amh_group", ["low", "normal", "high"]
        )

    if "follicle_size_band" in gdf.columns:
        gdf.expect_column_values_to_not_be_null("follicle_size_band")
        gdf.expect_column_distinct_values_to_be_in_set(
            "follicle_size_band", ["<12", "12-19", ">=20"]
        )

    for col in ["high_response_proxy", "ohss_risk_proxy"]:
        if col in gdf.columns:
            gdf.expect_column_values_to_be_in_set(col, [0, 1])

    # ---------- Validate and save result ----------
    result = gdf.validate()
    print("Validation success:", result["success"])

    # Convert to JSON‑serializable dict
    result_dict = result.to_json_dict()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2)

    print(f"Saved validation report to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
