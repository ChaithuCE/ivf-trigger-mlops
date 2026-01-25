from feast import FeatureView, Field, FileSource
from feast.types import Float32, Int64
from datetime import timedelta
from .entities import patient

ivf_offline_source = FileSource(
    path="data/trigger_day_prediction.parquet",
    timestamp_field="event_timestamp",
)

ivf_trigger_features = FeatureView(
    name="ivf_trigger_features",
    entities=[patient],
    ttl=timedelta(days=1),
    schema=[
        Field(name="Age", dtype=Int64),
        Field(name="AMH (ng/mL)", dtype=Float32),
        Field(name="Day", dtype=Int64),
        Field(name="Avg_Follicle_Size_mm", dtype=Float32),
        Field(name="Follicle_Count", dtype=Int64),
        Field(name="Estradiol_pg_mL", dtype=Float32),
        Field(name="Progesterone_ng_mL", dtype=Float32),
    ],
    online=True,
    source=ivf_offline_source,
)
