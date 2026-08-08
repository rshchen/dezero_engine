import numpy as np
from dezero.core import Variable
data = np.array(1.0)
x = Variable(data)

print(x.data)  # 輸出: 1.0