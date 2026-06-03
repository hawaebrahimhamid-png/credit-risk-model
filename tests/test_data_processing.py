import pandas as pd

def test_project_setup():
    assert True

def test_rfm_columns_exist():
    df = pd.DataFrame({
        "Recency": [1],
        "Frequency": [2],
        "Monetary": [100]
    })

    assert "Recency" in df.columns
    assert "Frequency" in df.columns
    assert "Monetary" in df.columns