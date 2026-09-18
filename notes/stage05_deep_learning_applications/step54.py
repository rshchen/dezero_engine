# 檔案：notes/stage04_deep_learning_from_scratch/step54.py
import numpy as np
import pytest
from dezero import Variable
from dezero.core import test_mode as dezero_test_mode
import dezero.functions as F


def test_dropout_in_test_mode():
  x = Variable(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))

  # 在測試模式下，Dropout 應直接透傳輸入
  with dezero_test_mode():
    y = F.dropout(x, dropout_ratio=0.5)

  np.testing.assert_array_equal(y.data, x.data)


def test_inverted_dropout_forward_scale():
  # 使用足夠大的樣本以檢驗機率分佈與期望值
  n_elements = 10000
  x = Variable(np.ones((1, n_elements), dtype=np.float32))
  dropout_ratio = 0.4
  keep_ratio = 1.0 - dropout_ratio

  y = F.dropout(x, dropout_ratio=dropout_ratio)

  # 驗證被遮蔽的神經元輸出為 0
  zeros_count = np.sum(y.data == 0.0)
  assert np.isclose(zeros_count / n_elements, dropout_ratio, atol=0.03)

  # 驗證未被遮蔽的神經元被 Inverted 放大為 1 / keep_ratio
  expected_scaled_val = 1.0 / keep_ratio
  active_vals = y.data[y.data != 0.0]
  np.testing.assert_allclose(active_vals, expected_scaled_val, rtol=1e-5)


def test_dropout_backward():
  x = Variable(np.ones((10, 10), dtype=np.float32))
  y = F.dropout(x, dropout_ratio=0.5)
  y.backward()

  # 反向傳播後，gx 應只在當初被保留的位置有值，且數值同比例縮放
  assert x.grad is not None
  expected_gx = (y.data != 0.0) * (1.0 / 0.5)
  np.testing.assert_allclose(x.grad.data, expected_gx, rtol=1e-5)

