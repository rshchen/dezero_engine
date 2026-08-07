import numpy as np
from dezero.core import Variable, Square, Exp
x = Variable(np.array(0.5))

A = Square()
B = Exp()
C = Square()

a = A(x)
b = B(a)
y = C(b)

print(y.data)

