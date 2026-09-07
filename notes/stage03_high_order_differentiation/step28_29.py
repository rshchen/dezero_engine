# 檔案：notes/stage03_high_order_differentiation/step28_29.py
import numpy as np
import pytest
from dezero import Variable


def rosenbrock(x0: Variable, x1: Variable) -> Variable:
  # y = 100 * (x1 - x0^2)^2 + (1 - x0)^2
  y = 100 * (x1 - x0**2) ** 2 + (1 - x0) ** 2
  return y


def test_newton_quadratic_single_step():
  # 1. 驗證二次函數單步收斂：y = (x - 2)^2, y' = 2(x - 2), y'' = 2
  x = Variable(np.array(5.0))
  y = (x - 2.0) ** 2

  # 一階導數計算
  y.backward()
  gx = x.grad

  # 解析二階導數值為 2.0
  g2x = 2.0

  # 牛頓法更新式：x_new = x - y' / y''
  x_new = x.data - gx / g2x

  # 二次函數應單步收斂至極值點 x = 2.0
  np.testing.assert_allclose(x_new, 2.0)


def test_newton_iterative_polynomial():
  # 2. 驗證多項式極小值迭代收斂：y = x^4 - 2x^2，在 x > 0 的極小點為 x = 1.0
  # 解析一階導數：y' = 4x^3 - 4x
  # 解析二階導數：y'' = 12x^2 - 4
  x_val = 2.0

  for _ in range(10):
    x = Variable(np.array(x_val))
    y = x**4 - 2 * (x**2)
    y.backward()

    gx = x.grad
    # 二階導數數值計算
    g2x = 12 * (x_val**2) - 4.0

    # 執行牛頓法迭代
    x_val = x_val - gx / g2x

  np.testing.assert_allclose(x_val, 1.0, atol=1e-5)


def test_rosenbrock_gradient_at_minimum():
  # 3. 驗證 Rosenbrock 函數在極小值點 (1.0, 1.0) 處的梯度為 0
  x0 = Variable(np.array(1.0))
  x1 = Variable(np.array(1.0))
  y = rosenbrock(x0, x1)

  y.backward()

  np.testing.assert_allclose(y.data, 0.0)
  np.testing.assert_allclose(x0.grad, 0.0)
  np.testing.assert_allclose(x1.grad, 0.0)

