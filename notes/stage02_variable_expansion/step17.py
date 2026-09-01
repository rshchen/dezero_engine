import weakref
import numpy as np
from dezero.core import Variable, square


# 驗證弱引用重構後求導計算正確性
def test_weakref_backward():
  x = Variable(np.array(3.0))
  y = square(square(x))  # y = (x^2)^2 = x^4
  y.backward()

  # dy/dx = 4 * x^3 = 4 * (3^3) = 108.0
  np.testing.assert_allclose(x.grad, np.array(108.0))


# 驗證循環參照解除後的即時記憶體回收
def test_memory_release():
  x = Variable(np.array(2.0))
  a = square(x)
  y = square(a)

  # 針對中間節點 a 建立弱引用觀察指標
  weak_a = weakref.ref(a)

  assert weak_a() is not None

  # 移除中間變數標籤與輸出變數標籤
  a = None
  y = None

  # 3. 驗證中間節點 a 在外部標籤移除後已被即時釋放（回傳 None）
  assert weak_a() is None

