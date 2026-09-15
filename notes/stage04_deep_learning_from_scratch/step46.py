# 檔案：notes/stage04_deep_learning_from_scratch/step46.py
import numpy as np
import pytest
from dezero import Variable
import dezero.functions as F
from dezero.models import MLP
from dezero.optimizers import Momentum


def test_optimizer_integration():
  np.random.seed(0)
  x_data = np.random.rand(100, 1)
  y_data = np.sin(2 * np.pi * x_data) + np.random.rand(100, 1)

  x = Variable(x_data)
  y = Variable(y_data)

  model = MLP((10, 1))
  optimizer = Momentum(lr=0.02, momentum=0.9)

  # 將最佳化目標模型綁定至 optimizer
  optimizer.setup(model)

  iters = 5000
  initial_loss = None

  for i in range(iters):
    y_pred = model(x)
    loss = F.mean_squared_error(y_pred, y)

    if i == 0:
      initial_loss = float(loss.data)

    model.cleargrads()
    loss.backward()

    # 統一由 optimizer 執行參數更新
    optimizer.update()

  final_loss = float(loss.data)

  assert final_loss < initial_loss
  assert final_loss < 0.25

