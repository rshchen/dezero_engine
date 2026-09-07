import numpy as np
from dezero.core import Variable, add, square
from dezero.utils import gradient_check


# 世代繼承與大小比較測試
def test_generation_hierarchy():
  x = Variable(np.array(2.0))
  a = square(x)
  y = add(square(a), square(a))

  assert x.generation == 0  # 葉節點世代為 0
  assert a.generation == 1  # 第一層輸出世代為 1
  assert y.generation == 3  # 第二層輸出世代為 2 + 1 = 3


# 跨層依賴複合計算圖求導測試：y = add(square(x), square(x)) => y = 2 * x^2 => dy/dx = 4x
def test_complex_dag_backward():
  x = Variable(np.array(2.0))
  a = square(x)
  y = add(square(a), square(a))  # y = a^2 + a^2 = (x^2)^2 + (x^2)^2 = 2 * x^4
  y.backward()

  # dy/dx = 8 * x^3 = 8 * (2^3) = 64
  np.testing.assert_allclose(x.grad, np.array(64.0))


# 梯度檢查工具自動化驗證跨層計算圖
def test_complex_dag_gradient_check():
  x = Variable(np.array(1.5))

  def f(x: Variable) -> Variable:
    # 構造分支長度不對稱的計算圖：y = (x^2)^2 + x^2
    a = square(x)
    return add(square(a), a)

  gradient_check(f, x)

