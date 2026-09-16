# 檔案：notes/stage04_deep_learning_from_scratch/step49.py
import math
import numpy as np
import pytest
from dezero import Variable
import dezero.datasets as D
import dezero.functions as F
from dezero.models import MLP
from dezero.optimizers import SGD
from dezero.transforms import Compose


def test_dataset_interface_and_transform():
  train_set = D.Spiral(train=True)
  assert len(train_set) == 300
  x_sample, t_sample = train_set[0]
  assert x_sample.shape == (2,)
  assert isinstance(int(t_sample), int)

  transform = Compose([
      lambda x: x * 2.0,
      lambda x: x + 1.0,
  ])
  transformed_set = D.Spiral(train=True, transform=transform)
  x_transformed, _ = transformed_set[0]
  np.testing.assert_allclose(x_transformed, x_sample * 2.0 + 1.0)


def test_dataset_minibatch_training():
  max_epoch = 300
  batch_size = 30
  hidden_size = 10
  lr = 1.0

  train_set = D.Spiral(train=True)
  data_size = len(train_set)
  max_iter = math.ceil(data_size / batch_size)

  model = MLP((hidden_size, 3))
  optimizer = SGD(lr=lr).setup(model)

  initial_loss = None

  for epoch in range(max_epoch):
    index = np.random.permutation(data_size)
    sum_loss = 0.0

    for i in range(max_iter):
      batch_index = index[i * batch_size : (i + 1) * batch_size]
      batch = [train_set[idx] for idx in batch_index]
      batch_x = np.array([example[0] for example in batch])
      batch_t = np.array([example[1] for example in batch])

      x = Variable(batch_x)
      t = Variable(batch_t)

      logits = model(x)
      loss = F.softmax_cross_entropy(logits, t)

      model.cleargrads()
      loss.backward()
      optimizer.update()

      sum_loss += float(loss.data) * len(t.data)

    avg_loss = sum_loss / data_size

    if epoch == 0:
      initial_loss = avg_loss

  assert avg_loss < initial_loss
  assert avg_loss < 0.2

