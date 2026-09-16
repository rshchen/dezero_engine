# 檔案：dezero/datasets.py
import numpy as np


def get_spiral(train=True):
  seed = 1984 if train else 2020
  np.random.seed(seed=seed)

  num_data = 100
  num_class = 3
  input_dim = 2

  data_size = num_data * num_class
  x = np.zeros((data_size, input_dim), dtype=np.float32)
  t = np.zeros(data_size, dtype=np.int32)

  for c in range(num_class):
    for i in range(num_data):
      r = i / num_data
      theta = c * 4.0 + 4.0 * r + np.random.randn() * 0.2
      ix = i + c * num_data
      x[ix] = np.array([r * np.sin(theta), r * np.cos(theta)]).flatten()
      t[ix] = c

  # 洗牌以破壞類別連續性
  indices = np.random.permutation(data_size)
  x = x[indices]
  t = t[indices]

  return x, t

