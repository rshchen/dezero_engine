import numpy as np
from dezero.core import Variable, Square

x = Variable(np.array(10.0))
f = Square()
y = f(x)

print(type(y))  # 輸出: <class '__main__.Variable'>
print(y.data)  # 輸出: 100.0