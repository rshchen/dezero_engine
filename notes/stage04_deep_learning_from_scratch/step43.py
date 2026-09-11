# 檔案：notes/stage04_deep_learning_from_scratch/step43.py
import numpy as np
import pytest
from dezero import Variable
import dezero.functions as F


def test_mlp_nonlinear_fitting():
  np.random.seed(0)
  # 生成 100 個非線性分佈樣本
  x_data = np.random.rand(100, 1)
  y_data = np.sin(2 * np.pi * x_data) + np.random.rand(100, 1)

  x = Variable(x_data)
  y = Variable(y_data)

  # 網路超參數設定
  in_size = 1
  hidden_size = 10
  out_size = 1

  # 固定小標準差隨機初始化 (0.01)
  W1 = Variable(0.01 * np.random.randn(in_size, hidden_size))
  b1 = Variable(np.zeros(hidden_size))
  W2 = Variable(0.01 * np.random.randn(hidden_size, out_size))
  b2 = Variable(np.zeros(out_size))

  def predict(x_input: Variable) -> Variable:
    # 第一層線性變換接非線性激活函數
    h = F.sigmoid(F.linear(x_input, W1, b1))
    # 第二層線性變換產出預測
    y_output = F.linear(h, W2, b2)
    return y_output

  lr = 0.2
  iters = 10000
  initial_loss = None
  final_loss = None

  for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y_pred, y)

    if i == 0:
      initial_loss = float(loss.data)

    # 必須清空所有四個可學習參數的梯度
    W1.cleargrad()
    b1.cleargrad()
    W2.cleargrad()
    b2.cleargrad()

    loss.backward()

    # 原地梯度更新
    W1.data -= lr * W1.grad.data
    b1.data -= lr * b1.grad.data
    W2.data -= lr * W2.grad.data
    b2.data -= lr * b2.grad.data

  final_loss = float(loss.data)

  # 驗證損失顯著降低，模型學會非線性空間映射
  assert final_loss < initial_loss
  assert final_loss < 0.25

