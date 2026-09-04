from typing import Callable, Set
import numpy as np
from dezero.core import Function, Variable, as_array
from pathlib import Path
import subprocess

def numerical_diff(f: Callable[[Variable], Variable], x: Variable, eps: float = 1e-4) -> np.ndarray:
    x0 = Variable(as_array(x.data - eps))
    x1 = Variable(as_array(x.data + eps))
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

# 通用梯度檢查工具函式
def gradient_check(
    f: Callable[[Variable], Variable],
    x: Variable,
    rtol: float = 1e-7,
    atol: float = 1e-5,
) -> None:

  # 進入前清空殘留梯度，避免閉包變數殘留值被累加
  x.cleargrad()
  # 1. 執行前向傳播與反向傳播
  y = f(x)
  y.backward()

  # 2. 計算數值微分
  num_grad = numerical_diff(f, x)

  # 3. 斷言比對解析梯度與數值梯度 (傳入相對與絕對誤差上限)
  np.testing.assert_allclose(x.grad, num_grad, rtol=rtol, atol=atol)


def _dot_var(v: Variable, verbose: bool = False) -> str:
  # 變數節點樣式：淺橘色填充，全面採用 f-string
  name = "" if v.name is None else v.name
  if verbose and v.data is not None:
    if v.name is not None:
      name += ": "
    name += f"{v.shape} {v.dtype}"

  return f'{id(v)} [label="{name}", color=orange, style=filled]\n'


def _dot_func(f: Function) -> str:
  # 函式節點樣式：矩形、淺藍色填充
  txt = (
      f"{id(f)} [label=\"{f.__class__.__name__}\", color=lightblue, style=filled,"
      ' shape=box]\n'
  )

  for x in f.inputs:
    txt += f"{id(x)} -> {id(f)}\n"
  for y in f.outputs:
    # 注意：outputs 存放的是 weakref，需解引用取得 Variable
    txt += f"{id(f)} -> {id(y())}\n"
  return txt


def get_dot_graph(output: Variable, verbose: bool = True) -> str:
  txt = ""
  funcs = []
  seen_set = set()

  def add_func(f: Function):
    if f not in seen_set:
      funcs.append(f)
      seen_set.add(f)

  # 從輸出的創作者開始往回走訪
  if output.creator is not None:
    add_func(output.creator)
  txt += _dot_var(output, verbose)

  while funcs:
    func = funcs.pop()
    txt += _dot_func(func)
    for x in func.inputs:
      txt += _dot_var(x, verbose)
      if x.creator is not None:
        add_func(x.creator)

  return "digraph g {\n" + txt + "}"


def plot_dot_graph(
    output: Variable,
    verbose: bool = True,
    to_file: str = "graph.png",
) -> None:
  dot_graph = get_dot_graph(output, verbose)

  # 1. 透過 pathlib 建立 ~/.dezero 目錄並寫入檔案
  tmp_dir = Path.home() / ".dezero"
  tmp_dir.mkdir(parents=True, exist_ok=True)
  graph_path = tmp_dir / "tmp_graph.dot"

  graph_path.write_text(dot_graph, encoding="utf-8")

  # 2. 擷取副檔名並調用 dot 指令
  to_path = Path(to_file)
  extension = to_path.suffix.lstrip(".")  # 取得副檔名屬性，例如 '.png' -> 'png'
  cmd = f"dot {graph_path} -T {extension} -o {to_path}"
  subprocess.run(cmd, shell=True, check=True)