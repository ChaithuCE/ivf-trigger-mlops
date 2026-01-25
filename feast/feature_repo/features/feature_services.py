from feast import FeatureService
from .feature_views import ivf_trigger_features

ivf_trigger_service_v1 = FeatureService(
    name="ivf_trigger_service_v1",
    features=[ivf_trigger_features],
    tags={"stage": "dev"},
)
