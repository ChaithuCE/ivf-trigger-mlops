from feast import Entity

patient = Entity(
    name="patient_id",
    join_keys=["Patient_ID"],  # column in your CSV
    description="IVF patient identifier",
)
