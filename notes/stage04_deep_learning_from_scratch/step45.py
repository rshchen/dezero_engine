# 檔案：notes/stage04_deep_learning_from_scratch/step45.py
import numpy as np
import pytest
from dezero import Variable
import dezero.functions as F
from dezero.models import MLP


def test_mlp_model_fitting():
  np.random.seed(0)
  # 生成 100 個非線性分佈樣本
  x_data = np.random.rand(100, 1)
  y_data = np.sin(2 * np.pi * x_data) + np.random.rand(100, 1)

  x = Variable(x_data)
  y = Variable(y_data)

  # 定義二層神經網路：隱藏層 10 維、輸出層 1 維
  model = MLP((10, 1))

  lr = 0.2
  iters = 10000
  initial_loss = None
  final_loss = None

  for i in range(iters):
    # 頂層容器直接執行前向計算
    y_pred = model(x)
    loss = F.mean_squared_error(y_pred, y)

    if i == 0:
      initial_loss = float(loss.data)

    # 批次清空所有子層的參數梯度
    model.cleargrads()

    loss.backward()

    # 透過生成器遞迴走訪模型內部所有的 Parameter
    for p in model.params():
      p.data -= lr * p.grad.data

  final_loss = float(loss.data)

  # 驗證整體模型順利收斂
  assert final_loss < initial_loss
  assert final_loss < 0.25

