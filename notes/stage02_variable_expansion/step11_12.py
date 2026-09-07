import numpy as np
from dezero.core import Variable, add


def test_add_forward():
  x0 = Variable(np.array(2.0))
  x1 = Variable(np.array(3.0))
  y = add(x0, x1)
  expected = np.array(5.0)
  assert y.data == expected


def test_add_output_type():
  x0 = Variable(np.array(1.5))
  x1 = Variable(np.array(2.5))
  y = add(x0, x1)
  assert isinstance(y, Variable)


def test_add_with_array():
  x0 = Variable(np.array([1.0, 2.0, 3.0]))
  x1 = Variable(np.array([4.0, 5.0, 6.0]))
  y = add(x0, x1)
  expected = np.array([5.0, 7.0, 9.0])
  np.testing.assert_allclose(y.data, expected)

