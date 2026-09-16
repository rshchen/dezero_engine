# 檔案：notes/stage04_deep_learning_from_scratch/step47.py
import numpy as np
import pytest
from dezero import Variable
import dezero.functions as F
from dezero.models import MLP
from dezero.optimizers import Momentum


def test_getitem_backward():
  x = Variable(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
  # 透過中括號切取所有列、第 1 與第 2 行特徵
  y = x[:, 1:3]
  loss = y.sum()
  loss.backward()

  expected_gx = np.array([[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
  np.testing.assert_array_equal(x.grad.data, expected_gx)


def test_softmax_cross_entropy_fitting():
  np.random.seed(0)
  # 生成三分類可分樣本：每類 20 個樣本
  x0 = np.random.randn(20, 2) + np.array([-2, -2])
  x1 = np.random.randn(20, 2) + np.array([2, -2])
  x2 = np.random.randn(20, 2) + np.array([0, 2])
  x_data = np.vstack([x0, x1, x2])
  t_data = np.array([0] * 20 + [1] * 20 + [2] * 20)

  x = Variable(x_data)
  t = Variable(t_data)

  model = MLP((10, 3))
  optimizer = Momentum(lr=0.1, momentum=0.9).setup(model)

  iters = 1000
  initial_loss = None

  for i in range(iters):
    # 前向計算得到未正規化的 Logits
    logits = model(x)
    # 使用算子融合的 Softmax 交叉熵
    loss = F.softmax_cross_entropy(logits, t)

    if i == 0:
      initial_loss = float(loss.data)

    model.cleargrads()
    loss.backward()
    optimizer.update()

  final_loss = float(loss.data)

  assert final_loss < initial_loss
  assert final_loss < 0.2

