# 檔案：notes/stage03_high_order_differentiation/step27.py
import math
from pathlib import Path
import numpy as np
import pytest
from dezero import Variable, sin
from dezero.utils import plot_dot_graph


# 2. 泰勒級數逼近實作
def my_sin(x: Variable, threshold: float = 1e-4) -> Variable:
  y = 0.0
  for i in range(100000):
    c = (-1) ** i / math.factorial(2 * i + 1)
    t = c * x ** (2 * i + 1)
    y = y + t
    if abs(t.data) < threshold:
      break
  return y


def test_primitive_sin():
  # 驗證原生 Sin 算子前向與反向計算
  x_val = np.pi / 4
  x = Variable(np.array(x_val), name="x")
  y = sin(x)
  y.backward()

  # sin(pi/4) == sqrt(2)/2, cos(pi/4) == sqrt(2)/2
  np.testing.assert_allclose(y.data, np.sin(x_val))
  np.testing.assert_allclose(x.grad, np.cos(x_val))


def test_taylor_sin_precision():
  # 驗證泰勒展開式逼近精度
  x_val = np.pi / 4
  x = Variable(np.array(x_val), name="x")
  y = my_sin(x, threshold=1e-4)
  y.backward()

  np.testing.assert_allclose(y.data, np.sin(x_val), atol=1e-4)
  np.testing.assert_allclose(x.grad, np.cos(x_val), atol=1e-4)


def test_plot_graphs_sandbox(tmp_path: Path):
  # 自動化沙盒測試：驗證兩種模式皆能順利輸出圖檔
  x1 = Variable(np.array(np.pi / 4), name="x")
  y1 = sin(x1)
  img1 = tmp_path / "primitive.png"
  plot_dot_graph(y1, verbose=False, to_file=str(img1))
  assert img1.is_file()
  assert img1.stat().st_size > 0

  x2 = Variable(np.array(np.pi / 4), name="x")
  y2 = my_sin(x2, threshold=1e-3)
  img2 = tmp_path / "taylor.png"
  plot_dot_graph(y2, verbose=False, to_file=str(img2))
  assert img2.is_file()
  assert img2.stat().st_size > 0


def test_manual_visual_inspection_workflow():
  # 本機手動檢驗工作流：產出實體圖檔以供比對拓撲結構複雜度
  x1 = Variable(np.array(np.pi / 4), name="x")
  y1 = sin(x1)
  file_primitive = Path("sin_primitive_check.png")
  plot_dot_graph(y1, verbose=False, to_file=str(file_primitive))
  assert file_primitive.is_file()

  x2 = Variable(np.array(np.pi / 4), name="x")
  y2 = my_sin(x2, threshold=1e-3)
  file_taylor = Path("sin_taylor_check.png")
  plot_dot_graph(y2, verbose=False, to_file=str(file_taylor))
  assert file_taylor.is_file()

