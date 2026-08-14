import numpy as np
from dezero.core import Variable, add
from dezero.utils import gradient_check


# 多輸入加法反向傳播偏微分驗證 (y = x0 + x1 => dy/dx0 = 1, dy/dx1 = 1)
def test_add_backward():
  x0 = Variable(np.array(2.0))
  x1 = Variable(np.array(3.0))
  y = add(x0, x1)
  y.backward()

  assert x0.grad == np.array(1.0)
  assert x1.grad == np.array(1.0)


# 重複變數累加與 cleargrad 測試 (y = x + x => dy/dx = 2)
def test_same_variable_and_cleargrad():
  x = Variable(np.array(3.0))
  y = add(x, x)
  y.backward()

  # 驗證梯度是否正確累加為 2.0 而非被覆蓋為 1.0
  assert x.grad == np.array(2.0)

  # 清除舊梯度
  x.cleargrad()
  assert x.grad is None

  # 再次計算新圖 y2 = x + x + x (dy/dx = 3)
  y2 = add(add(x, x), x)
  y2.backward()
  assert x.grad == np.array(3.0)


# 結合梯度檢查工具驗證加法算子
def test_add_gradient_check():
  x = Variable(np.array(2.0))

  def f(x: Variable) -> Variable:
    return add(x, Variable(np.array(3.0)))

  gradient_check(f, x)

