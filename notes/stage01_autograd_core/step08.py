import numpy as np
from dezero.core import Exp, Square, Variable
from dezero.utils import numerical_diff

# 1. 前向傳播 (Define-by-Run 動態建立計算圖)
x = Variable(np.array(0.5))
A, B, C = Square(), Exp(), Square()

a = A(x)
b = B(a)
y = C(b)

# 2. 自動化反向傳播 (觸發堆疊迴圈圖走訪)
y.backward()

def f(x: Variable) -> Variable:
  return C(B(A(x)))


num_grad = numerical_diff(f, x)
print(f'數值微分 (num_grad): {num_grad}')
print(f'絕對誤差           : {np.abs(x.grad - num_grad)}')