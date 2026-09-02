import numpy as np
from dezero.core import Variable


def test_variable_properties():
  raw_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  x = Variable(raw_data, name="x_matrix")

  # 1. 驗證屬性直接存取（無括號）
  assert x.shape == (2, 3)
  assert x.ndim == 2
  assert x.size == 6
  assert x.dtype == np.float64
  assert x.name == "x_matrix"

  # 2. 驗證全域函式 len() 協定轉發
  assert len(x) == 2


def test_variable_repr():
  x = Variable(np.array([1, 2, 3]))
  # 3. 驗證 repr 格式
  expected = "variable([1 2 3])"
  assert str(x) == expected

