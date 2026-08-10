import numpy as np
from dezero.core import Variable, exp, square
from dezero.utils import numerical_diff

# 1. 前向傳播 (使用 Step 09 快捷函式簡化複合運算)
x = Variable(np.array(0.5))
y = square(exp(square(x)))

# 2. 自動化反向傳播 (觸發堆疊迴圈圖走訪)
y.backward()

# 3. 數值微分對照與誤差檢驗
def f(x: Variable) -> Variable:
    return square(exp(square(x)))

num_grad = numerical_diff(f, x)
print(f'自動求導梯度 (x.grad) : {x.grad}')
# print(f'數值微分 (num_grad): {num_grad}')
# print(f'絕對誤差           : {np.abs(x.grad - num_grad)}')

