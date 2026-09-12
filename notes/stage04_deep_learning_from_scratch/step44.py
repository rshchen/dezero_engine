# 檔案：notes/stage04_deep_learning_from_scratch/step44.py
import numpy as np
import pytest
from dezero import Variable
import dezero.functions as F
from dezero.layers import Linear


def test_layer_parameter_management_fitting():
  np.random.seed(0)
  # 生成 100 個非線性分佈樣本
  x_data = np.random.rand(100, 1)
  y_data = np.sin(2 * np.pi * x_data) + np.random.rand(100, 1)

  x = Variable(x_data)
  y = Variable(y_data)

  # 物件導向定義兩個線性層（隱藏層 10 維，輸出層 1 維）
  l1 = Linear(10)
  l2 = Linear(1)

  def predict(x_input: Variable) -> Variable:
    # 第一層前向計算，搭配 sigmoid 激活函數
    h = F.sigmoid(l1(x_input))
    # 第二層前向計算產出預測值
    y_output = l2(h)
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

    # 批次清空各層所註冊的全部參數梯度
    l1.cleargrads()
    l2.cleargrads()

    loss.backward()

    # 使用生成器遍歷該層的所有 Parameter 實例進行原地權重更新
    for p in l1.params():
      p.data -= lr * p.grad.data
    for p in l2.params():
      p.data -= lr * p.grad.data

  final_loss = float(loss.data)

  # 驗證損失顯著降低，模型參數透過 Layer 管理順利更新收斂
  assert final_loss < initial_loss
  assert final_loss < 0.25

