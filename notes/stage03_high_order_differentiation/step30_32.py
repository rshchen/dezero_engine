# 檔案：notes/stage03_high_order_differentiation/step32.py
import numpy as np
import pytest
from dezero import Variable, sin


def test_sin_higher_order_derivatives():
  # 驗證 sin(x) 的高階循環導數
  x_val = np.pi / 4
  x = Variable(np.array(x_val))
  y = sin(x)

  # 一階導數：y' = cos(x)
  y.backward(create_graph=True)
  gx = x.grad
  np.testing.assert_allclose(gx.data, np.cos(x_val))

  # 二階導數：y'' = -sin(x)
  x.cleargrad()
  gx.backward(create_graph=True)
  g2x = x.grad
  np.testing.assert_allclose(g2x.data, -np.sin(x_val))

  # 三階導數：y''' = -cos(x)
  x.cleargrad()
  g2x.backward()
  g3x = x.grad
  np.testing.assert_allclose(g3x.data, -np.cos(x_val))


def test_automated_newton_method_polynomial():
  # 驗證牛頓法數值收斂：y = x^4 - 2x^2 在 x > 0 的極小值點為 x = 1.0
  x = Variable(np.array(2.0))
  iters = 10

  for _ in range(iters):
    y = x**4 - 2 * (x**2)

    # 計算一階導數並保留圖結構
    x.cleargrad()
    y.backward(create_graph=True)
    gx = x.grad

    # 對一階導數反向傳播求得二階導數
    x.cleargrad()
    gx.backward()
    g2x = x.grad

    # 牛頓法更新式：x = x - f'(x) / f''(x)
    x.data -= gx.data / g2x.data

  np.testing.assert_allclose(x.data, 1.0, atol=1e-5)

