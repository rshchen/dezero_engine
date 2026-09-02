import numpy as np
from dezero.core import Config, Variable, no_grad, square


# 驗證反向傳播中間與終端梯度即時釋放
def test_retain_grad():
  x = Variable(np.array(2.0))
  a = square(x)
  y = square(a)

  # 執行反向傳播，預設 retain_grad=False
  y.backward(retain_grad=False)

  # 葉節點梯度必須保留
  assert x.grad is not None
  # 中間節點 a 與終端節點 y 的梯度皆被即時回收為 None
  assert a.grad is None
  assert y.grad is None


# 驗證 no_grad 推論模式不建構計算圖
def test_no_grad_context():
  assert Config.enable_backprop is True

  with no_grad():
    x = Variable(np.array(2.0))
    y = square(x)
    # 推論模式下，算子不記錄血緣關係
    assert y.creator is None

  # 驗證離開 with 區塊後，全域組態安全還原
  assert Config.enable_backprop is True

