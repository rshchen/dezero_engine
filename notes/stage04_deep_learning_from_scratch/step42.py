# 檔案：notes/stage04_deep_learning_from_scratch/step42.py
import numpy as np
import pytest
from dezero import Variable
from dezero.functions import mean_squared_error


def test_mean_squared_error_forward_backward():
  x0_data = np.array([[1.0], [2.0], [3.0]])
  x1_data = np.array([[1.5], [1.5], [3.5]])

  x0 = Variable(x0_data)
  x1 = Variable(x1_data)

  # 計算均方誤差
  loss = mean_squared_error(x0, x1)

  # 前向檢驗：diff = [-0.5, 0.5, -0.5], diff^2 = [0.25, 0.25, 0.25], sum/3 = 0.25
  np.testing.assert_allclose(loss.data, 0.25)

  # 執行反向傳播
  loss.backward()

  # 反向檢驗：gx0 = (2 / N) * (x0 - x1)
  N = len(x0_data)
  expected_gx0 = (2.0 / N) * (x0_data - x1_data)
  expected_gx1 = -expected_gx0

  np.testing.assert_allclose(x0.grad.data, expected_gx0)
  np.testing.assert_allclose(x1.grad.data, expected_gx1)


def test_linear_regression_gradient_descent():
  np.random.seed(0)
  x_data = np.random.rand(100, 1)
  y_data = 2.0 * x_data + 5.0 + 0.1 * np.random.randn(100, 1)

  x = Variable(x_data)
  y = Variable(y_data)

  W = Variable(np.zeros((1, 1)))
  b = Variable(np.zeros((1,)))

  def predict(x_var: Variable) -> Variable:
    return x_var @ W + b

  lr = 0.1
  iters = 100
  initial_loss = None
  final_loss = None

  for i in range(iters):
    y_pred = predict(x)
    loss = mean_squared_error(y_pred, y)

    if i == 0:
      initial_loss = float(loss.data)

    W.cleargrad()
    b.cleargrad()
    loss.backward()

    # In-place 更新權重與偏差
    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data

  final_loss = float(loss.data)

  # 驗證損失大幅下降且參數接近目標值 (W ≈ 2.0, b ≈ 5.0)
  assert final_loss < initial_loss
  assert final_loss < 0.1
  np.testing.assert_allclose(W.data[0, 0], 2.0, atol=0.3)
  np.testing.assert_allclose(b.data[0], 5.0, atol=0.3)

