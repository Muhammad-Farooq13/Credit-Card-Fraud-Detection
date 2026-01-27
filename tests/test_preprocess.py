import os
import numpy as np
import pandas as pd

from src.data.preprocess import preprocess_dataset


def test_preprocess_outputs(tmp_path):
    df = pd.DataFrame({"Time": [0, 1, 2, 3], "Amount": [10, 20, 30, 40], "Class": [0, 0, 1, 1]})
    artifacts = preprocess_dataset(df, tmp_path)

    assert os.path.exists(artifacts["train"])
    data = np.load(artifacts["train"])
    assert "X" in data and "y" in data
