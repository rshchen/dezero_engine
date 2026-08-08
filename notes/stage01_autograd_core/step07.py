import numpy as np
from typing import Optional
from dezero.core import Exp, Square, Variable
from dezero.utils import numerical_diff








# --- 實機執行測試 ---
# 1. 建立輸入變數與算子實體
x = Variable(np.array(0.5))
A, B, C = Square(), Exp(), Square()

# 2. 前向傳播 (Define-by-Run 動態組裝計算圖)
a = A(x)
b = B(a)
y = C(b)

# 3. 觸發自動遞迴求導
y.backward()

print(f'遞迴自動求導結果 (x.grad) : {x.grad}')

# 數值微分驗證
def f(x: Variable) -> Variable:
  return C(B(A(x)))
num_grad = numerical_diff(f, x)
print(f'數值微分 (num_grad): {num_grad}')
print(f'絕對誤差           : {np.abs(x.grad - num_grad)}')