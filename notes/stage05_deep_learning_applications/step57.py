# 檔案：notes/stage05_deep_learning_applications/step57.py
import numpy as np
import pytest
from dezero import Variable
from dezero.datasets import MNIST
from dezero.dataloaders import DataLoader
import dezero.functions as F
import dezero.layers as L
from dezero.optimizers import Adam
from dezero.models import Model


def test_conv2d_forward_shape_and_value():
  x_data = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
  x = Variable(x_data)

  W_data = np.ones((1, 1, 3, 3), dtype=np.float32)
  W = Variable(W_data)

  y = F.conv2d(x, W, b=None, stride=1, pad=0)

  assert y.shape == (1, 1, 2, 2)
  np.testing.assert_almost_equal(y.data[0, 0, 0, 0], 45.0)


def test_pooling_forward_max():
  x_data = np.array(
      [[[[1, 2, 5, 6], [3, 4, 7, 8], [9, 10, 13, 14], [11, 12, 15, 16]]]],
      dtype=np.float32,
  )
  x = Variable(x_data)

  y = F.pooling(x, kernel_size=2, stride=2, pad=0)

  assert y.shape == (1, 1, 2, 2)
  expected_out = np.array([[[[4, 8], [12, 16]]]], dtype=np.float32)
  np.testing.assert_array_equal(y.data, expected_out)


def test_conv2d_backward():
  x = Variable(np.random.randn(2, 3, 8, 8).astype(np.float32))
  conv = L.Conv2d(out_channels=4, kernel_size=3, stride=1, pad=1)

  y = conv(x)
  loss = y.sum()
  loss.backward()

  assert x.grad is not None
  assert conv.W.grad is not None
  assert conv.b.grad is not None


class TinyCNN(Model):

  def __init__(self):
    super().__init__()
    self.conv = L.Conv2d(out_channels=4, kernel_size=3, stride=1, pad=1)
    self.fc = L.Linear(10)

  def forward(self, x):
    h = F.relu(self.conv(x))
    h = F.pooling(h, kernel_size=2, stride=2)
    h = F.reshape(h, (h.shape[0], -1))
    return self.fc(h)


def test_cnn_mnist_fast_overfit():
  train_set = MNIST(train=True)
  loader = DataLoader(train_set, batch_size=10, shuffle=False)
  x, t = next(loader)
  x = x / 255.0

  model = TinyCNN()
  optimizer = Adam(lr=0.02).setup(model)

  losses = []
  for _ in range(20):
    y = model(x)
    loss = F.softmax_cross_entropy(y, t)
    model.cleargrads()
    loss.backward()
    optimizer.update()
    losses.append(float(loss.data))

  assert losses[-1] < losses[0] * 0.1

