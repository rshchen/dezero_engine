# 檔案：dezero/transforms.py
class Compose:

  def __init__(self, transforms):
    self.transforms = transforms

  def __call__(self, data):
    for f in self.transforms:
      data = f(data)
    return data


class Normalize:

  def __init__(self, mean=0.0, std=1.0):
    self.mean = mean
    self.std = std

  def __call__(self, array):
    return (array - self.mean) / self.std


class Flatten:

  def __call__(self, array):
    return array.flatten()

