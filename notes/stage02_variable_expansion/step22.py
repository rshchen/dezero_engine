import numpy as np
from dezero.core import Variable


def test_neg_operator():
  x = Variable(np.array(2.5))
  # 測試一元負號
  y = -x
  np.testing.assert_allclose(y.data, -2.5)

  y.backward()
  # dy/dx = -1.0
  np.testing.assert_allclose(x.grad, -1.0)


def test_mixed_arithmetic():
  x = Variable(np.array(2.0))

  y1 = x + 3.0
  y2 = 3.0 + x  # 觸發 __radd__
  y3 = 3.0 * x  # 觸發 __rmul__
  y4 = 5.0 - x  # 觸發 __rsub__
  y5 = 4.0 / x  # 觸發 __rtruediv__

  np.testing.assert_allclose(y1.data, 5.0)
  np.testing.assert_allclose(y2.data, 5.0)
  np.testing.assert_allclose(y3.data, 6.0)
  np.testing.assert_allclose(y4.data, 3.0)
  np.testing.assert_allclose(y5.data, 2.0)

  # 測試 rsub 反向傳播: y = 5.0 - x -> dy/dx = -1.0
  y4.backward()
  np.testing.assert_allclose(x.grad, -1.0)


def test_complex_formula():
  x = Variable(np.array(3.0))
  # 測試複合公式: y = -(x^2) + 2x + 1
  y = -(x**2) + 2.0 * x + 1.0

  # -(9) + 6 + 1 = -2.0
  np.testing.assert_allclose(y.data, -2.0)

  # dy/dx = -2x + 2 -> 在 x = 3.0 時為 -2*3 + 2 = -4.0
  y.backward()
  np.testing.assert_allclose(x.grad, -4.0)

