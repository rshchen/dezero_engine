import numpy as np
from dezero import Variable
from dezero.utils import gradient_check


# 測試函數宣告
def sphere(x: Variable, y: Variable) -> Variable:
  return x**2 + y**2


def matyas(x: Variable, y: Variable) -> Variable:
  return 0.26 * (x**2 + y**2) - 0.48 * x * y


def goldstein_price(x: Variable, y: Variable) -> Variable:
  m = 1 + (x + y + 1) ** 2 * (
      19 - 14 * x + 3 * x**2 - 14 * y + 6 * x * y + 3 * y**2
  )
  n = 30 + (2 * x - 3 * y) ** 2 * (
      18 - 32 * x + 12 * x**2 + 48 * y - 36 * x * y + 27 * y**2
  )
  return m * n


def test_sphere_gradient():
  x = Variable(np.array(2.0))
  y = Variable(np.array(3.0))

  gradient_check(lambda var: sphere(var, y), x)
  gradient_check(lambda var: sphere(x, var), y)


def test_matyas_gradient():
  x = Variable(np.array(1.5))
  y = Variable(np.array(2.0))

  gradient_check(lambda var: matyas(var, y), x)
  gradient_check(lambda var: matyas(x, var), y)


def test_goldstein_price_gradient():
  x = Variable(np.array(0.5))
  y = Variable(np.array(-0.5))

  gradient_check(lambda var: goldstein_price(var, y), x, rtol=1e-3, atol=1e-3)
  gradient_check(lambda var: goldstein_price(x, var), y, rtol=1e-3, atol=1e-3)


def test_matyas_optimization():
  x = Variable(np.array(1.0))
  y = Variable(np.array(1.0))
  lr = 0.1
  iters = 600

  for _ in range(iters):
    z = matyas(x, y)

    # 1. 每次求導前必須清空舊梯度
    x.cleargrad()
    y.cleargrad()

    # 2. 反向傳播計算梯度
    z.backward()

    # 3. 沿負梯度方向更新參數資料
    x.data -= lr * x.grad
    y.data -= lr * y.grad

  # 驗證迭代後收斂至理論極小值 (0.0, 0.0) 附近
  np.testing.assert_allclose(x.data, 0.0, atol=0.1)
  np.testing.assert_allclose(y.data, 0.0, atol=0.1)

