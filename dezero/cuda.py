# 檔案：dezero/cuda.py
import numpy as np

# 嘗試安全匯入 cupy
gpu_enable = True
try:
  import cupy as cp
  cupy = cp
except ImportError:
  gpu_enable = False


def get_array_module(x):
  """依據輸入陣列的型別，回傳對應的模組 (numpy 或 cupy)"""
  if not gpu_enable:
    return np
  # cupy.get_array_module 可根據傳入物件回傳 numpy 或 cupy
  return cp.get_array_module(x)


def as_numpy(x):
  """將輸入轉換為 numpy.ndarray"""
  if isinstance(x, np.ndarray):
    return x
  if np.isscalar(x):
    return np.array(x)
  if isinstance(x, cp.ndarray):
    return x.get()  # CuPy 專屬方法，將資料從 GPU 取回 CPU
  return np.array(x)


def as_cupy(x):
  """將輸入轉換為 cupy.ndarray"""
  if isinstance(x, cp.ndarray):
    return x
  if isinstance(x, np.ndarray) or np.isscalar(x):
    if not gpu_enable:
      raise Exception("CuPy is not installed.")
    return cp.asarray(x)  # 將資料傳送至 GPU
  return cp.asarray(x)

