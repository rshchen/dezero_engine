# 檔案：dezero/datasets.py
import numpy as np

from pathlib import Path
from sklearn.datasets import fetch_openml


class Dataset:

  def __init__(self, train=True, transform=None, target_transform=None):
    self.train = train
    self.transform = transform
    self.target_transform = target_transform
    if self.transform is None:
      self.transform = lambda x: x
    if self.target_transform is None:
      self.target_transform = lambda x: x

    self.data = None
    self.label = None
    self.prepare()

  def __getitem__(self, index):
    assert np.isscalar(index), "Dataset 只支援單一索引存取"
    if self.label is None:
      return self.transform(self.data[index])

    return (
        self.transform(self.data[index]),
        self.target_transform(self.label[index]),
    )

  def __len__(self):
    return len(self.data)

  def prepare(self):
    pass

class Spiral(Dataset):

    def prepare(self):
        self.data, self.label = get_spiral(self.train)

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


class MNIST(Dataset):

  def prepare(self):
    cache_dir = Path.home() / ".dezero"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "mnist.npz"

    if not cache_file.exists():
      mnist = fetch_openml(
          "mnist_784", version=1, as_frame=False, parser="liac-arff"
      )
      data = mnist.data.reshape(-1, 1, 28, 28).astype(np.float32)
      label = mnist.target.astype(np.int64)
      np.savez_compressed(cache_file, data=data, label=label)

    raw_data = np.load(cache_file)
    data = raw_data["data"]
    label = raw_data["label"]

    if self.train:
      self.data = data[:60000]
      self.label = label[:60000]
    else:
      self.data = data[60000:]
      self.label = label[60000:]