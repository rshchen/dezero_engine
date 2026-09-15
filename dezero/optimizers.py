# 檔案：dezero/optimizers.py
from __future__ import annotations


class Optimizer:

    def __init__(self):
        self.target = None
        self.hooks = []

    def setup(self, target):
        self.target = target
        return self

    def update(self):
        # 收集所有非 None 梯度的參數
        params = [p for p in self.target.params() if p.grad is not None]

        # 執行前置處理（如 Weight Decay、Gradient Clipping）
        for f in self.hooks:
            f(params)

        # 對所有參數進行具體數值更新
        for param in params:
            self.update_one(param)

    def update_one(self, param):
        raise NotImplementedError

    def add_hook(self, f):
        self.hooks.append(f)

class SGD(Optimizer):

    def __init__(self, lr: float = 0.01):
        super().__init__()
        self.lr = lr

    def update_one(self, param):
        param.data -= self.lr * param.grad.data

import numpy as np


class Momentum(Optimizer):

    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.vs = {}

    def update_one(self, param):
        # 依據參數記憶體位址 (id) 維護獨立的歷史狀態
        param_id = id(param)
        if param_id not in self.vs:
            self.vs[param_id] = np.zeros_like(param.data)

        v = self.vs[param_id]
        v *= self.momentum
        v -= self.lr * param.grad.data
        param.data += v

class AdaGrad(Optimizer):

    def __init__(self, lr: float = 0.001, eps: float = 1e-8):
        super().__init__()
        self.lr = lr
        self.eps = eps
        self.hs = {}

    def update_one(self, param):
        param_id = id(param)
        if param_id not in self.hs:
            self.hs[param_id] = np.zeros_like(param.data)

        h = self.hs[param_id]
        grad = param.grad.data

        h += grad * grad
        param.data -= self.lr * grad / (np.sqrt(h) + self.eps)

class Adam(Optimizer):

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.t = 0
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.ms = {}
        self.vs = {}

    def update(self):
        self.t += 1
        super().update()

    def update_one(self, param):
        param_id = id(param)
        if param_id not in self.ms:
            self.ms[param_id] = np.zeros_like(param.data)
            self.vs[param_id] = np.zeros_like(param.data)

        m = self.ms[param_id]
        v = self.vs[param_id]
        grad = param.grad.data

        m += (1 - self.beta1) * (grad - m)
        v += (1 - self.beta2) * (grad * grad - v)

        # 偏差修正合併後的有效步長
        fix1 = 1.0 - self.beta1**self.t
        fix2 = 1.0 - self.beta2**self.t
        lr_t = self.lr * np.sqrt(fix2) / fix1

        param.data -= lr_t * m / (np.sqrt(v) + self.eps)

