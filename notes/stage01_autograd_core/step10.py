import numpy as np
from dezero.core import Variable, exp, square
from dezero.utils import gradient_check, numerical_diff


# 1. 前向傳播正確性測試
def test_square_forward():
  x = Variable(np.array(2.0))
  y = square(x)
  expected = np.array(4.0)
  assert y.data == expected


# 2. 單一算子反向傳播測試 (使用 dezero.utils 匯入的 gradient_check)
def test_square_backward():
  x = Variable(np.array(3.0))
  gradient_check(square, x)


# 3. 複合算子梯度流測試 (square -> exp -> square)
def test_gradient_flow():
  x = Variable(np.array(0.5))

  def f(x: Variable) -> Variable:
    return square(exp(square(x)))

  gradient_check(f, x)



