# 檔案：notes/stage04_deep_learning_from_scratch/step51.py
import numpy as np
import pytest
from dezero import Variable
import dezero.datasets as D
import dezero.functions as F
from dezero.dataloaders import DataLoader
from dezero.models import MLP
from dezero.optimizers import SGD
import dezero.transforms as T


def test_relu_forward_and_backward():
  x = Variable(np.array([[-1.0, 0.0], [2.0, -3.0]]))
  y = F.relu(x)
  np.testing.assert_array_equal(y.data, np.array([[0.0, 0.0], [2.0, 0.0]]))

  y.backward()
  expected_gx = np.array([[0.0, 0.0], [1.0, 0.0]])
  np.testing.assert_array_equal(x.grad.data, expected_gx)


def test_mnist_dataset_pipeline():
  transform = T.Compose([
      T.Flatten(),
      T.Normalize(mean=0.0, std=255.0),
  ])
  dataset = D.MNIST(train=True, transform=transform)

  assert len(dataset) == 60000
  x, t = dataset[0]

  assert x.shape == (784,)
  assert np.min(x) >= 0.0 and np.max(x) <= 1.0
  assert isinstance(int(t), int)


def test_mnist_mlp_training_convergence():
  max_epoch = 10
  batch_size = 30
  lr = 0.2

  transform = T.Compose([
      T.Flatten(),
      T.Normalize(mean=0.0, std=255.0),
  ])
  full_train_set = D.MNIST(train=True, transform=transform)
  full_test_set = D.MNIST(train=False, transform=transform)

  # 取子集以維持測試套件的快速執行
  full_train_set.data = full_train_set.data[:600]
  full_train_set.label = full_train_set.label[:600]
  full_test_set.data = full_test_set.data[:200]
  full_test_set.label = full_test_set.label[:200]

  train_loader = DataLoader(full_train_set, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(full_test_set, batch_size=batch_size, shuffle=False)

  model = MLP((100, 10), activation=F.relu)
  optimizer = SGD(lr=lr).setup(model)

  final_train_loss = None
  final_train_acc = None
  final_test_acc = None

  for epoch in range(max_epoch):
    sum_loss = 0.0
    sum_acc = 0.0

    # 1. 訓練階段
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

    final_train_loss = sum_loss / len(full_train_set)
    final_train_acc = sum_acc / len(full_train_set)

    # 2. 測試/評估階段
    sum_test_acc = 0.0
    for x_batch, t_batch in test_loader:
      x = Variable(x_batch)
      t = Variable(t_batch)
      y = model(x)
      acc = F.accuracy(y, t)
      sum_test_acc += float(acc.data) * len(t_batch)

    final_test_acc = sum_test_acc / len(full_test_set)

    print(
        f"epoch {epoch + 1} | train_loss: {final_train_loss:.4f} | "
        f"train_acc: {final_train_acc:.4f} | test_acc: {final_test_acc:.4f}"
    )

  assert final_train_loss < 0.5
  assert final_train_acc > 0.8
  assert final_test_acc > 0.8

