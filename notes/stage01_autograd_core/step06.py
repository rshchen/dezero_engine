import numpy as np
from dezero.core import Exp, Square, Variable
from dezero.utils import numerical_diff

# x->A->a->B->b->C->y
# 1. 前向傳播
x = Variable(np.array(0.5))
A, B, C = Square(), Exp(), Square()
a = A(x)
b = B(a)
y = C(b)

# 2. 設定終點 y 的梯度並進行手動反向傳播
y.grad = np.array(1.0)
b.grad = C.backward(y.grad)
a.grad = B.backward(b.grad)
x.grad = A.backward(a.grad)

print(f'解析微分 (x.grad) : {x.grad}')


# 3. 數值微分驗證
def f(x: Variable) -> Variable:
  return C(B(A(x)))


num_grad = numerical_diff(f, x)
print(f'數值微分 (num_grad): {num_grad}')
print(f'絕對誤差           : {np.abs(x.grad - num_grad)}')

