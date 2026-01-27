import numpy as np

from src.models.train import train_model


def test_train_creates_model(tmp_path):
    X = np.random.randn(100, 3)
    y = np.array([0] * 90 + [1] * 10)
    train_path = tmp_path / "train.npz"
    np.savez(train_path, X=X, y=y)

    config = {"train_path": str(train_path), "output_path": str(tmp_path / "model.joblib")}
    model_path = train_model(config)
    assert tmp_path.joinpath("model.joblib").exists()
    assert model_path == str(tmp_path / "model.joblib")
