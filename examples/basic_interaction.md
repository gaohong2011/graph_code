# 基本交互示例

这是 Graph Code 的基本交互示例。

## 1. 列出目录内容

```
You: 列出当前目录的所有文件和文件夹

Graph Code: 当前目录包含以下内容：

**目录：**
- `.git/` - Git 仓库数据
- `.venv/` - Python 虚拟环境
- `examples/` - 示例目录
- `graph_code/` - 主要代码目录

**文件：**
- `.env.example` (259B)
- `README.md` (4KB)
- `requirements.txt` (312B)
```

## 2. 读取文件

```
You: 读取 README.md 的前 30 行

Graph Code: 这是 README.md 的前 30 行：

    1 | # Graph Code
    2 |
    3 | 基于 LangGraph 的 AI 编程助手...
    ...
```

## 3. 搜索代码

```
You: 搜索所有 Python 文件中定义函数的代码行

Graph Code: 找到 25 个函数定义：

**graph_code/main.py:**
  25: def print_banner(console: Console):
  42: def setup_config(args) -> bool:
  85: def handle_pending_interaction(state, console: Console):

**graph_code/agent/graph.py:**
  19: def build_agent() -> StateGraph:
  57: def run_agent(user_input: str, ...):
```

## 4. 分析代码结构

```
You: 分析 graph_code/tools/ 目录的结构

Graph Code: `graph_code/tools/` 目录包含以下工具模块：

1. **file_tools.py** - 文件操作工具
   - read_file: 读取文件内容
   - write_file: 写入文件
   - list_directory: 列出目录
   - glob_search: 文件搜索

2. **code_tools.py** - 代码分析工具
   - grep_search: 正则搜索
   - read_code_chunk: 读取代码片段

3. **exec_tools.py** - 执行工具
   - bash_command: Bash 命令
   - python_execute: Python 执行

4. **interaction.py** - 人机交互
   - ask_user: 提问
   - confirm_action: 确认操作
```

## 5. 执行命令

```
You: 运行 pwd 和 ls -la 命令

Graph Code: 命令执行结果：

Command: pwd
Exit code: 0
STDOUT:
/Users/gaohong/go/src/graph_code

Command: ls -la
Exit code: 0
STDOUT:
total 24
drwxr-xr-x  7 user staff  224 Mar 25 22:00 .
drwxr-xr-x  3 user staff   96 Mar 25 21:50 ..
-rw-r--r--  1 user staff  259 Mar 25 22:00 .env.example
...
```

## 6. 组合操作

```
You: 先搜索所有包含 "class " 的 Python 文件，
    然后读取第一个类所在的文件

Graph Code: 搜索中...
找到 3 个类定义：

1. graph_code/config.py:8 - class Config:
2. graph_code/tools/interaction.py:10 - class InteractionStore:

读取 graph_code/config.py：

```python
class Config:
    """Graph Code configuration."""

    def __init__(self):
        self.llm_api_key: Optional[str] = os.getenv("LLM_API_KEY")
        ...
```
```
