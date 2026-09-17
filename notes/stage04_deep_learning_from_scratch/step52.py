# 檔案：notes/stage05_deep_learning_applications/step52.py
import numpy as np
import pytest
from dezero import Variable
from dezero.models import MLP
import dezero.cuda as cuda


def test_cuda_fallback_and_get_array_module():
  # 驗證在 Apple Silicon / 無 CUDA 環境下的優雅降級
  assert cuda.gpu_enable is False

  x = np.array([1.0, 2.0, 3.0])
  xp = cuda.get_array_module(x)
  assert xp is np


def test_variable_to_cpu_and_to_gpu_fallback():
  x_data = np.array([1.0, 2.0, 3.0])
  v = Variable(x_data)

  # 測試轉回 CPU 維持正常
  v.to_cpu()
  assert isinstance(v.data, np.ndarray)
  np.testing.assert_array_equal(v.data, x_data)

  # 測試在無 CuPy 環境下轉向 GPU 應丟出例外
  with pytest.raises(Exception):
    v.to_gpu()


def test_layer_params_to_cpu():
  model = MLP((10, 3))

  # 模擬前向傳播以動態初始化各層參數權重
  x = Variable(np.random.randn(2, 4))
  _ = model(x)

  # 呼叫模型批次轉移
  model.to_cpu()

  # 檢查內部所有參數是否皆維持在 CPU 上
  for param in model.params():
    assert isinstance(param.data, np.ndarray)

