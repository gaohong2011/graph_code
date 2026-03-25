# 代码生成示例

展示 Graph Code 如何生成和修改代码。

## 1. 创建简单脚本

```
You: 创建一个 fibonacci.py，实现斐波那契数列计算

Graph Code: 已创建 fibonacci.py：

```python
"""
斐波那契数列计算模块
"""


def fibonacci(n: int) -> int:
    """
    计算第 n 个斐波那契数

    Args:
        n: 斐波那契数列的索引（从0开始）

    Returns:
        第 n 个斐波那契数

    Raises:
        ValueError: 如果 n 为负数
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return n

    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def fibonacci_sequence(count: int) -> list[int]:
    """
    生成前 count 个斐波那契数

    Args:
        count: 要生成的斐波那契数数量

    Returns:
        包含斐波那契数的列表
    """
    return [fibonacci(i) for i in range(count)]


if __name__ == "__main__":
    # 示例：打印前 10 个斐波那契数
    print("前 10 个斐波那契数:")
    for i, num in enumerate(fibonacci_sequence(10)):
        print(f"F({i}) = {num}")
```

让我验证代码：
```
$ python fibonacci.py
前 10 个斐波那契数:
F(0) = 0
F(1) = 1
F(2) = 1
F(3) = 2
F(4) = 3
F(5) = 5
F(6) = 8
F(7) = 13
F(8) = 21
F(9) = 34
```

✓ 代码运行成功！
```

## 2. 添加测试

```
You: 为 fibonacci.py 添加单元测试

Graph Code: 已创建 test_fibonacci.py：

```python
"""
斐波那契数列的单元测试
"""

import pytest
from fibonacci import fibonacci, fibonacci_sequence


def test_fibonacci_base_cases():
    """测试基本情况"""
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1


def test_fibonacci_small_numbers():
    """测试小数字"""
    assert fibonacci(2) == 1
    assert fibonacci(3) == 2
    assert fibonacci(4) == 3
    assert fibonacci(5) == 5


def test_fibonacci_larger_number():
    """测试较大数字"""
    assert fibonacci(10) == 55
    assert fibonacci(20) == 6765


def test_fibonacci_negative():
    """测试负数输入"""
    with pytest.raises(ValueError):
        fibonacci(-1)


def test_fibonacci_sequence():
    """测试序列生成"""
    result = fibonacci_sequence(5)
    assert result == [0, 1, 1, 2, 3]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

运行测试：
```
$ python -m pytest test_fibonacci.py -v
============================= test session starts ==============================
test_fibonacci.py::test_fibonacci_base_cases PASSED
test_fibonacci.py::test_fibonacci_small_numbers PASSED
test_fibonacci.py::test_fibonacci_larger_number PASSED
test_fibonacci.py::test_fibonacci_negative PASSED
test_fibonacci.py::test_fibonacci_sequence PASSED
============================== 5 passed in 0.01s ===============================
```

✓ 所有测试通过！
```

## 3. 修改现有代码

```
You: 读取 fibonacci.py，添加一个递归实现版本

Graph Code: 已为 fibonacci.py 添加递归实现：

```python
def fibonacci_recursive(n: int, memo: dict = None) -> int:
    """
    使用递归和记忆化计算第 n 个斐波那契数

    Args:
        n: 斐波那契数列的索引
        memo: 记忆化字典（内部使用）

    Returns:
        第 n 个斐波那契数
    """
    if memo is None:
        memo = {}

    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return n
    if n in memo:
        return memo[n]

    memo[n] = fibonacci_recursive(n - 1, memo) + fibonacci_recursive(n - 2, memo)
    return memo[n]
```

已更新文件！
```

## 4. 创建实用工具

```
You: 创建一个 HTTP 客户端工具类，支持 GET 和 POST 请求

Graph Code: 已创建 http_client.py：

```python
"""
HTTP 客户端工具类
"""

import json
from typing import Any, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError
from urllib.parse import urlencode


class HttpClient:
    """简单的 HTTP 客户端"""

    def __init__(self, base_url: str = "", timeout: int = 30):
        """
        初始化 HTTP 客户端

        Args:
            base_url: 基础 URL
            timeout: 请求超时时间（秒）
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers: dict[str, str] = {
            "User-Agent": "GraphCode-HttpClient/1.0"
        }

    def get(self, path: str, params: Optional[dict] = None) -> dict:
        """
        发送 GET 请求

        Args:
            path: 请求路径
            params: URL 参数

        Returns:
            响应数据
        """
        url = f"{self.base_url}/{path.lstrip('/')}"
        if params:
            url += "?" + urlencode(params)

        return self._request(url, method="GET")

    def post(self, path: str, data: Optional[dict] = None) -> dict:
        """
        发送 POST 请求

        Args:
            path: 请求路径
            data: 请求体数据

        Returns:
            响应数据
        """
        url = f"{self.base_url}/{path.lstrip('/')}"
        return self._request(url, method="POST", data=data)

    def _request(self, url: str, method: str = "GET",
                 data: Optional[dict] = None) -> dict:
        """发送 HTTP 请求"""
        headers = self.headers.copy()

        if data:
            body = json.dumps(data).encode("utf-8")
            headers["Content-Type"] = "application/json"
        else:
            body = None

        try:
            req = Request(url, data=body, headers=headers, method=method)
            with urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except URLError as e:
            raise Exception(f"HTTP request failed: {e}")


# 使用示例
if __name__ == "__main__":
    client = HttpClient("https://api.github.com")

    # GET 请求
    response = client.get("/users/octocat")
    print(f"User: {response.get('login')}")
    print(f"Bio: {response.get('bio')}")
```

让我测试这个客户端：
```
$ python http_client.py
User: octocat
Bio: Hi, I'm the Octocat!
```

✓ HTTP 客户端工作正常！
```

## 5. 代码模板生成

```
You: 创建一个 Python 项目模板，包含：
1. 基本的项目结构
2. 配置文件支持
3. 日志设置
4. 主入口文件

Graph Code: 已创建项目模板：

my_project/
├── my_project/
│   ├── __init__.py
│   ├── __main__.py      # 入口点
│   ├── config.py        # 配置管理
│   ├── logger.py        # 日志设置
│   └── main.py          # 主逻辑
├── tests/
│   ├── __init__.py
│   └── test_main.py     # 测试
├── config.yaml          # 配置文件
├── requirements.txt     # 依赖
└── README.md           # 说明文档

主要文件内容：
...
```
