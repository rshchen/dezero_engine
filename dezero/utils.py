from typing import Callable
import numpy as np
from dezero.core import Variable, as_array

def numerical_diff(f: Callable[[Variable], Variable], x: Variable, eps: float = 1e-4) -> np.ndarray:
    x0 = Variable(as_array(x.data - eps))
    x1 = Variable(as_array(x.data + eps))
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

# 通用梯度檢查工具函式
def gradient_check(
    f: Callable[[Variable], Variable],
    x: Variable,
    rtol: float = 1e-7,
    atol: float = 1e-5,
) -> None:

  # 進入前清空殘留梯度，避免閉包變數殘留值被累加
  x.cleargrad()
  # 1. 執行前向傳播與反向傳播
  y = f(x)
  y.backward()

  # 2. 計算數值微分
  num_grad = numerical_diff(f, x)

  # 3. 斷言比對解析梯度與數值梯度 (傳入相對與絕對誤差上限)
  np.testing.assert_allclose(x.grad, num_grad, rtol=rtol, atol=atol)
