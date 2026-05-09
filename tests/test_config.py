"""Smoke-тест: конфиги читаются и `extends` корректно мерджится."""
from src.utils.io import load_config


def test_base_config_loads():
    cfg = load_config("base.yaml")
    assert cfg["universe"]["top_n"] == 1500
    assert cfg["rebalance"]["lag_trading_days"] == 21
    assert len(cfg["features"]["macd_pairs"]) == 8
    assert cfg["seeds"] == [0, 1, 2, 3, 4]


def test_mlp_reg_extends_base():
    cfg = load_config("mlp_reg.yaml")

    assert cfg["universe"]["top_n"] == 1500
    assert cfg["rebalance"]["lag_trading_days"] == 21

    assert cfg["model"]["name"] == "mlp_reg"
    assert cfg["model"]["loss"] == "smooth_l1"
    assert cfg["training"]["lr"] == 1e-3


def test_mlp_clf_dropout_overrides():
    cfg = load_config("mlp_clf.yaml")
    assert cfg["model"]["name"] == "mlp_clf"
    assert cfg["model"]["dropout"] == 0.5
    assert cfg["inference"]["mc_dropout_samples"] == 30


def test_lstm_seq_len():
    cfg = load_config("lstm_reg.yaml")
    assert cfg["model"]["seq_len"] == 12
    assert cfg["model"]["hidden_size"] == 32
