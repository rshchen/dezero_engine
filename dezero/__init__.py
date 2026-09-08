# 檔案：dezero/__init__.py
from dezero.core import (
    Config,
    Function,
    Variable,
    as_array,
    as_variable,
    no_grad,
    setup_variable,
    using_config,
)

# 優先執行運算子掛載，確保 Variable 具備重載方法
setup_variable()

# 核心就緒後，再匯入擴充算子庫
from dezero.functions import cos, exp, sin, square, tanh

__all__ = [
    "Variable",
    "Function",
    "Config",
    "using_config",
    "no_grad",
    "as_array",
    "as_variable",
    "square",
    "exp",
    "sin",
    "cos",
    "tanh",
]

