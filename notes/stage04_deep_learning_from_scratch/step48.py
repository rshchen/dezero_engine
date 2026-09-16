# 檔案：notes/stage04_deep_learning_from_scratch/step48.py
import math
import numpy as np
import pytest
from dezero import Variable
import dezero.datasets as D
import dezero.functions as F
from dezero.models import MLP
from dezero.optimizers import SGD


def test_spiral_dataset_shape():
  x, t = D.get_spiral(train=True)

  assert x.shape == (300, 2)
  assert t.shape == (300,)
  assert set(np.unique(t)) == {0, 1, 2}


def test_spiral_minibatch_training():
  max_epoch = 300
  batch_size = 30
  hidden_size = 10
  lr = 1.0

  x_data, t_data = D.get_spiral(train=True)
  data_size = len(x_data)
  max_iter = math.ceil(data_size / batch_size)

  model = MLP((hidden_size, 3))
  optimizer = SGD(lr=lr).setup(model)

  initial_loss = None

  for epoch in range(max_epoch):
    index = np.random.permutation(data_size)
    sum_loss = 0.0

    for i in range(max_iter):
      batch_index = index[i * batch_size : (i + 1) * batch_size]
      batch_x = Variable(x_data[batch_index])
      batch_t = Variable(t_data[batch_index])

      logits = model(batch_x)
      loss = F.softmax_cross_entropy(logits, batch_t)

      model.cleargrads()
      loss.backward()
      optimizer.update()

      sum_loss += float(loss.data) * len(batch_t.data)

    avg_loss = sum_loss / data_size

    if epoch == 0:
      initial_loss = avg_loss

    if (epoch + 1) % 50 == 0:
      print(f"epoch {epoch + 1}, loss {avg_loss:.2f}")

  assert avg_loss < initial_loss
  assert avg_loss < 0.2

