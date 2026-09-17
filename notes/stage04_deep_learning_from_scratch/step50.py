# 檔案：notes/stage04_deep_learning_from_scratch/step50.py
import numpy as np
import pytest
from dezero import Variable
import dezero.datasets as D
import dezero.functions as F
from dezero.dataloaders import DataLoader
from dezero.models import MLP
from dezero.optimizers import SGD


def test_accuracy_function():
  y = np.array([
      [0.8, 0.1, 0.1],
      [0.2, 0.7, 0.1],
      [0.3, 0.6, 0.1],
  ])
  t = np.array([0, 1, 2])

  acc = F.accuracy(y, t)
  np.testing.assert_almost_equal(float(acc.data), 2.0 / 3.0)


def test_dataloader_training_with_accuracy():
  max_epoch = 300
  batch_size = 30
  hidden_size = 10
  lr = 1.0

  train_set = D.Spiral(train=True)
  train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

  model = MLP((hidden_size, 3))
  optimizer = SGD(lr=lr).setup(model)

  final_epoch_loss = None
  final_epoch_acc = None

  for epoch in range(max_epoch):
    sum_loss = 0.0
    sum_acc = 0.0

    for x_batch, t_batch in train_loader:
      x = Variable(x_batch)
      t = Variable(t_batch)

      y = model(x)
      loss = F.softmax_cross_entropy(y, t)
      acc = F.accuracy(y, t)

      model.cleargrads()
      loss.backward()
      optimizer.update()

      batch_len = len(t_batch)
      sum_loss += float(loss.data) * batch_len
      sum_acc += float(acc.data) * batch_len

    final_epoch_loss = sum_loss / len(train_set)
    final_epoch_acc = sum_acc / len(train_set)

    if (epoch + 1) % 50 == 0:
      print(f"epoch {epoch + 1} | loss: {final_epoch_loss:.4f} | acc: {final_epoch_acc:.4f}")

  assert final_epoch_loss < 0.2
  assert final_epoch_acc > 0.9

