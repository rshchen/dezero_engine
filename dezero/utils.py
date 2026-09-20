# 檔案：dezero/utils.py
from __future__ import annotations

from pathlib import Path
import subprocess
from typing import TYPE_CHECKING, Callable, Sequence, Set
import numpy as np

if TYPE_CHECKING:
  from dezero.core import Function, Variable
import dezero.cuda as cuda

# 純量轉型工具函式
def as_array(x, array_module=np):
    if np.isscalar(x):
        return array_module.array(x)
    return x


def numerical_diff(f: Callable[[Variable], Variable], x: Variable, eps: float = 1e-4) -> np.ndarray:
    from dezero.core import Variable  # 於呼叫時局部載入類別實體
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

def reshape_sum_backward(
        gy: Variable,
        x_shape: tuple[int, ...],
        axis: int | tuple[int, ...] | None,
        keepdims: bool,
    ) -> Variable:
  """將求和輸出的梯度重塑為與輸入維度數一致的形狀，以利後續廣播。"""
  import dezero.functions as F

  if len(x_shape) == 0 or keepdims:
    return gy

  actual_axis: set[int]
  if axis is None:
    actual_axis = set(range(len(x_shape)))
  elif isinstance(axis, int):
    actual_axis = {axis if axis >= 0 else axis + len(x_shape)}
  else:
    actual_axis = {a if a >= 0 else a + len(x_shape) for a in axis}

  target_shape = tuple(
      1 if i in actual_axis else dim for i, dim in enumerate(x_shape)
  )
  return F.reshape(gy, target_shape)

def sum_to_array(x: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
  """將 NumPy 陣列沿著廣播軸求和，壓縮至指定的 target_shape。"""
  # 找出當初因為維度數不足而被額外補上並拉長的維度
  ndim = len(target_shape)
  lead = x.ndim - ndim
  lead_axis = tuple(range(lead))

  # 找出長度為 1 且原本在 x 中大於 1 的軸（被廣播軸）
  axis = [i + lead for i, sx in enumerate(target_shape) if sx == 1]

  y = x.sum(axis=lead_axis + tuple(axis), keepdims=True)
  if lead > 0:
    y = y.squeeze(lead_axis)
  return y

def get_conv_outsize(input_size, filter_size, stride, pad):
  """計算卷積運算的空間輸出維度（Height 或 Width）。"""
  return (input_size + 2 * pad - filter_size) // stride + 1

def im2col(x, kernel_size, stride=1, pad=0, to_matrix=True):
  """將 4D 輸入張量依據滑動視窗展開為 2D 矩陣。"""
  x = np.asarray(x)
  N, C, H, W = x.shape
  KH, KW = (
      (kernel_size, kernel_size)
      if isinstance(kernel_size, int)
      else kernel_size
  )
  SH, SW = (stride, stride) if isinstance(stride, int) else stride
  PH, PW = (pad, pad) if isinstance(pad, int) else pad

  OH = get_conv_outsize(H, KH, SH, PH)
  OW = get_conv_outsize(W, KW, SW, PW)

  img = np.pad(
      x,
      [(0, 0), (0, 0), (PH, PH + SH - 1), (PW, PW + SW - 1)],
      mode="constant",
  )
  col = np.zeros((N, C, KH, KW, OH, OW), dtype=x.dtype)

  for j in range(KH):
    j_max = j + SH * OH
    for i in range(KW):
      i_max = i + SW * OW
      col[:, :, j, i, :, :] = img[:, :, j:j_max:SH, i:i_max:SW]

  if to_matrix:
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * OH * OW, -1)

  return col


def col2im(col, x_shape, kernel_size, stride=1, pad=0, to_matrix=True):
  """將展開的梯度矩陣沿空間視窗累加還原回 4D 張量形狀。"""
  N, C, H, W = x_shape
  KH, KW = (
      (kernel_size, kernel_size)
      if isinstance(kernel_size, int)
      else kernel_size
  )
  SH, SW = (stride, stride) if isinstance(stride, int) else stride
  PH, PW = (pad, pad) if isinstance(pad, int) else pad

  OH = get_conv_outsize(H, KH, SH, PH)
  OW = get_conv_outsize(W, KW, SW, PW)

  if to_matrix:
    col = col.reshape(N, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)

  img = np.zeros(
      (N, C, H + 2 * PH + SH - 1, W + 2 * PW + SW - 1), dtype=col.dtype
  )

  for j in range(KH):
    j_max = j + SH * OH
    for i in range(KW):
      i_max = i + SW * OW
      img[:, :, j:j_max:SH, i:i_max:SW] += col[:, :, j, i, :, :]

  return img[:, :, PH : H + PH, PW : W + PW]

