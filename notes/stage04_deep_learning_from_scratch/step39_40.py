# 檔案：notes/stage04_deep_learning_from_scratch/step39_40.py
import numpy as np
import pytest
from dezero import Variable
from dezero.functions import broadcast_to, sum, sum_to


def test_sum_forward_backward():
  x_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  x = Variable(x_data)

  # 全域求和
  y = sum(x)
  assert y.shape == ()

  # 反向求導驗證廣播填充
  y.backward()
  assert x.grad.shape == (2, 3)
  np.testing.assert_allclose(x.grad.data, np.ones((2, 3)))


def test_sum_axis_and_keepdims():
  x_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  x = Variable(x_data)

  # 沿 axis 0 求和且不保持維度
  y = sum(x, axis=0)
  assert y.shape == (3,)

  y.backward()
  assert x.grad.shape == (2, 3)
  np.testing.assert_allclose(x.grad.data, np.ones((2, 3)))


def test_broadcast_to_backward_duality():
  x_data = np.array([1.0, 2.0, 3.0])
  x = Variable(x_data)

  # 廣播擴展至 (2, 3)
  y = broadcast_to(x, (2, 3))
  assert y.shape == (2, 3)

  y.backward()
  # 沿著擴展維度累加，梯度值應為 2.0
  assert x.grad.shape == (3,)
  np.testing.assert_allclose(x.grad.data, np.full((3,), 2.0))


def test_add_broadcasting_automatic_differentiation():
  x0 = Variable(np.ones((2, 3)))
  x1 = Variable(np.ones((1, 3)))

  # 觸發原生算子多載廣播加法
  y = x0 + x1
  assert y.shape == (2, 3)

  y.backward()
  # x0 未擴展，梯度全為 1
  assert x0.grad.shape == (2, 3)
  np.testing.assert_allclose(x0.grad.data, np.ones((2, 3)))

  # x1 在 axis 0 擴展了 2 次，梯度應求和為 2
  assert x1.grad.shape == (1, 3)
  np.testing.assert_allclose(x1.grad.data, np.full((1, 3), 2.0))

