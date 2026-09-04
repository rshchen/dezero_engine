# 檔案：notes/stage03_high_order_differentiation/step26.py
from pathlib import Path
import subprocess
import numpy as np
import pytest
from dezero import Variable
from dezero.utils import _dot_func, _dot_var, get_dot_graph, plot_dot_graph


def sphere(x: Variable, y: Variable) -> Variable:
  return x**2 + y**2


def test_dot_var_formatting():
  # 1. 驗證單一變數轉譯 DOT 格式
  x = Variable(np.array(1.0), name="x")
  dot_str = _dot_var(x, verbose=False)

  # 斷言包含節點唯一識別碼與指定標籤名稱
  assert f'{id(x)} [label="x", color=orange, style=filled]' in dot_str


def test_sphere_computational_graph_dot():
  # 2. 驗證完整 Sphere 計算圖的 DOT 拓撲結構
  x = Variable(np.array(2.0), name="x")
  y = Variable(np.array(3.0), name="y")
  z = sphere(x, y)
  z.name = "z"

  dot_graph = get_dot_graph(z, verbose=False)

  # 驗證 digraph 根宣告
  assert dot_graph.startswith("digraph g {")
  assert dot_graph.endswith("}")

  # 驗證輸入與輸出節點均存在於圖中
  assert f'{id(x)} [label="x"' in dot_graph
  assert f'{id(y)} [label="y"' in dot_graph
  assert f'{id(z)} [label="z"' in dot_graph

  # 驗證 Add 算子節點存在且輸出指向 z
  add_func = z.creator
  assert f'{id(add_func)} [label="Add"' in dot_graph
  assert f"{id(add_func)} -> {id(z)}" in dot_graph

  # 驗證 x 的前向有向邊連接至對應的次方算子節點
  x_square_func = add_func.inputs[0].creator
  assert f"{id(x)} -> {id(x_square_func)}" in dot_graph


def test_plot_dot_graph_execution(tmp_path: Path):
  # 3. 常態自動化整合測試（使用 pytest 沙盒目錄）
  x = Variable(np.array(2.0), name="x")
  y = Variable(np.array(3.0), name="y")
  z = sphere(x, y)
  z.name = "z"

  output_image = tmp_path / "sphere.png"
  plot_dot_graph(z, verbose=False, to_file=str(output_image))

  # 斷言輸出圖檔確實建立，且檔案大小大於 0 位元組
  assert output_image.is_file()
  assert output_image.stat().st_size > 0


def test_manual_visual_inspection_workflow():
  # 4. 肉眼檢驗工作流：輸出本機實際圖檔並透過系統預覽工具開啟
  x = Variable(np.array(2.0), name="x")
  y = Variable(np.array(3.0), name="y")
  z = sphere(x, y)
  z.name = "z"

  output_file = Path("sphere_manual_check.png")
  plot_dot_graph(z, verbose=True, to_file=str(output_file))

  assert output_file.is_file()
  assert output_file.stat().st_size > 0

  # 自動調用 macOS 系統預覽程式開啟圖檔供開發者肉眼檢查
  # 終端機等價指令：open sphere_manual_check.png
  subprocess.run(f"open {output_file}", shell=True, check=True)

