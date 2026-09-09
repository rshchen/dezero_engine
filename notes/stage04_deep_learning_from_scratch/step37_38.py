# 檔案：notes/stage04_deep_learning_from_scratch/step37_38.py
import numpy as np
import pytest
from dezero import Variable
from dezero.functions import reshape, sin, transpose


def test_elementwise_tensor_gradient():
  x_data = np.array([[1.0, 2.0], [3.0, 4.0]])
  x = Variable(x_data)
  y = sin(x)

  # 執行反向傳播
  y.backward()

  # 驗證逐元素鏈鎖律梯度結果：dL/dx = cos(x)
  expected_gx = np.cos(x_data)
  np.testing.assert_allclose(x.grad.data, expected_gx)


def test_reshape_forward_backward():
  x = Variable(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
  # 呼叫 reshape 將 (2, 3) 轉換為 (6,)
  y = reshape(x, (6,))

  assert y.shape == (6,)

  # 驗證反向伴隨變換
  y.backward()
  assert x.grad.shape == (2, 3)
  np.testing.assert_allclose(x.grad.data, np.ones((2, 3)))


def test_transpose_general_permutation():
  # 輸入形狀：(2, 3, 4)
  x_data = np.arange(24).reshape((2, 3, 4)).astype(np.float64)
  x = Variable(x_data)

  axes = (1, 2, 0)
  # 執行高維軸置換
  y = transpose(x, axes=axes)
  assert y.shape == (3, 4, 2)

  # 驗證透過逆置換還原形狀
  y.backward()
  assert x.grad.shape == (2, 3, 4)
  np.testing.assert_allclose(x.grad.data, np.ones((2, 3, 4)))


def test_transpose_property_syntax():
  x = Variable(np.array([[1.0, 2.0], [3.0, 4.0]]))
  # 使用轉置快捷屬性
  y = x.T
  assert y.shape == (2, 2)
  np.testing.assert_allclose(y.data, np.array([[1.0, 3.0], [2.0, 4.0]]))

