# 檔案：notes/stage04_deep_learning_from_scratch/step41.py
import numpy as np
import pytest
from dezero import Variable
from dezero.functions import matmul


def test_matmul_forward_backward_shapes():
  x_data = np.random.randn(2, 3)
  w_data = np.random.randn(3, 4)

  x = Variable(x_data)
  W = Variable(w_data)

  # 前向計算
  y = matmul(x, W)
  assert y.shape == (2, 4)

  # 反向傳播
  y.backward()

  # 驗證梯度形狀嚴格符合原變數形狀
  assert x.grad.shape == (2, 3)
  assert W.grad.shape == (3, 4)

  # 驗證數值符合解析矩陣乘積公式
  gy_data = np.ones((2, 4))
  np.testing.assert_allclose(x.grad.data, gy_data.dot(w_data.T))
  np.testing.assert_allclose(W.grad.data, x_data.T.dot(gy_data))


def test_matmul_operator_overload():
  x_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  w_data = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

  x = Variable(x_data)
  W = Variable(w_data)

  # 觸發 @ 運算子多載
  y = x @ W

  assert y.shape == (2, 2)

  y.backward()
  assert x.grad.shape == (2, 3)
  assert W.grad.shape == (3, 2)

