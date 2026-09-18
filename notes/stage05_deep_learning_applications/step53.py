# 檔案：notes/stage04_deep_learning_from_scratch/step53.py
from pathlib import Path
import numpy as np
import pytest
from dezero import Variable
from dezero.models import MLP


def test_flatten_params_naming():
  # 建立兩層 MLP 模型：輸入維度由前向傳播自動推導，隱藏層 10，輸出 3
  model = MLP((10, 3))
  x = Variable(np.random.randn(2, 4).astype(np.float32))
  _ = model(x)  # 觸發參數延遲初始化

  # 準備空字典傳入收集
  params_dict = {}
  model._flatten_params(params_dict)

  # 驗證關鍵路徑鍵名存在
  expected_keys = {"l0/W", "l0/b", "l1/W", "l1/b"}
  assert expected_keys.issubset(params_dict.keys())

  # 驗證取得之物件均已具備數值
  for key in expected_keys:
    assert params_dict[key].data is not None


def test_model_save_and_load_weights(tmp_path):
  # 使用 pytest 提供的 tmp_path 建立臨時儲存路徑
  weight_path = Path(tmp_path) / "checkpoints" / "mlp_weights.npz"

  # 1. 建立並初始化原始模型
  model = MLP((10, 3))
  x = Variable(np.random.randn(5, 4).astype(np.float32))
  y_orig = model(x)

  # 2. 序列化權重至 .npz
  model.save_weights(weight_path)
  assert weight_path.exists()

  # 3. 建立全新模型架構實例並注入權重
  new_model = MLP((10, 3))
  _ = new_model(x)  # 確保新模型形狀已初始化
  new_model.load_weights(weight_path)

  # 4. 驗證新舊模型輸出與內部權重完全一致
  y_loaded = new_model(x)
  np.testing.assert_allclose(y_orig.data, y_loaded.data, rtol=1e-5, atol=1e-6)

  orig_params = {}
  model._flatten_params(orig_params)
  loaded_params = {}
  new_model._flatten_params(loaded_params)

  for key in orig_params:
    np.testing.assert_array_equal(orig_params[key].data, loaded_params[key].data)

