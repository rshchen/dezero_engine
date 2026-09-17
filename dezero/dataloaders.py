# 檔案：dezero/dataloaders.py
import math
import numpy as np
from dezero import cuda

class DataLoader:

  def __init__(self, dataset, batch_size, shuffle=True, gpu=False):
    self.dataset = dataset
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.data_size = len(dataset)
    self.max_iter = math.ceil(self.data_size / batch_size)
    self.reset()
    self.gpu = gpu

  def reset(self):
    self.iteration = 0
    if self.shuffle:
      self.index = np.random.permutation(self.data_size)
    else:
      self.index = np.arange(self.data_size)

  def __iter__(self):
    return self

  def __next__(self):
    if self.iteration >= self.max_iter:
      self.reset()
      raise StopIteration

    i = self.iteration
    batch_size = self.batch_size
    batch_index = self.index[i * batch_size : (i + 1) * batch_size]
    batch = [self.dataset[idx] for idx in batch_index]

    # 依據 gpu 標籤動態選用 CuPy 或 NumPy 直接組裝陣列
    xp = cuda.cupy if self.gpu else np
    x = xp.array([example[0] for example in batch])
    t = xp.array([example[1] for example in batch])

    self.iteration += 1
    return x, t

  def next(self):
    return self.__next__()

  def to_cpu(self):
    self.gpu = False

  def to_gpu(self):
    self.gpu = True

