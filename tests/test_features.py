from src.features import build_features
from src.ingest import generate_sample_dataset


def test_build_features_has_expected_columns():
    raw_df = generate_sample_dataset(n_rows=20, random_seed=7)
    processed_df = build_features(raw_df)
    assert "patient_id" in processed_df.columns
    assert "readmitted_30d" in processed_df.columns
    assert "age" in processed_df.columns
    assert processed_df.shape[0] == 20
