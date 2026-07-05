"""计算器工具:基于 AST 白名单的安全表达式求值(不用裸 eval)。"""
import ast
import math

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": (
            "计算数学表达式。支持四则运算、括号、** 幂、% 取余、// 整除,"
            "以及 math 函数:sqrt/sin/cos/tan/log/log2/log10/exp/floor/ceil,"
            "常量 pi/e,内置 round/abs/min/max/sum/pow。示例:(123+456)*789、sqrt(2026)"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "要计算的数学表达式"},
            },
            "required": ["expression"],
        },
    },
}

_FUNCS = {name: getattr(math, name) for name in
          ["sqrt", "sin", "cos", "tan", "asin", "acos", "atan",
           "log", "log2", "log10", "exp", "floor", "ceil", "factorial"]}
_FUNCS.update(round=round, abs=abs, min=min, max=max, sum=sum, pow=pow)
_CONSTS = {"pi": math.pi, "e": math.e}

_BINOPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
           ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
           ast.Pow: lambda a, b: a ** b, ast.Mod: lambda a, b: a % b,
           ast.FloorDiv: lambda a, b: a // b}
_UNARY = {ast.UAdd: lambda a: a, ast.USub: lambda a: -a}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id in _FUNCS and not node.keywords:
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_eval(e) for e in node.elts]
    raise ValueError(f"不支持的表达式成分: {ast.dump(node)[:60]}")


def run(args):
    expr = str(args.get("expression", "")).strip()
    if not expr:
        return "错误:expression 为空"
    try:
        tree = ast.parse(expr, mode="eval")
        result = _eval(tree)
    except Exception as e:
        return f"计算出错: {e}"
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return repr(result)
