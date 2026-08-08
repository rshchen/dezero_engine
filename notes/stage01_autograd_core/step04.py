import numpy as np
from dezero.core import Variable, Square, Exp
from dezero.utils import numerical_diff

def f(x: Variable) -> Variable:
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))

x = Variable(np.array(0.5))
dy = numerical_diff(f, x)

print(dy)  # 輸出: 3.29744...

