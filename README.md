# Graph Code

基于 LangGraph 的 AI 编程助手，类似 Claude Code，但仅需配置 API Key 即可使用。

## 特性

- **简单配置**: 仅需 API Key 和 Base URL，无需登录
- **LangGraph 驱动**: 使用状态机管理复杂任务流程
- **丰富工具**: 文件操作、代码搜索、命令执行、人机交互
- **安全优先**: 敏感操作默认需要确认
- **交互友好**: 漂亮的命令行界面

## 快速开始

### 1. 安装

```bash
# 克隆项目
cd graph_code

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置

**方式一：环境变量（推荐）**

```bash
export LLM_API_KEY=your_api_key
export LLM_BASE_URL=https://api.moonshot.cn/v1
export LLM_MODEL=moonshot-v1-8k
```

**方式二：命令行参数**

```bash
python -m graph_code \
  --api-key your_api_key \
  --base-url https://api.moonshot.cn/v1 \
  --model moonshot-v1-8k
```

**方式三：.env 文件**

```bash
cp .env.example .env
# 编辑 .env 填入你的配置
```

### 3. 运行

```bash
# 交互模式
python -m graph_code

# 执行单个命令
python -m graph_code "列出当前目录下的所有 Python 文件"

# 使用参数
python -m graph_code --model gpt-4 --working-dir /path/to/project
```

## 配置选项

| 环境变量 | 说明 | 默认值 | 示例 |
|---------|------|--------|------|
| `LLM_API_KEY` | LLM API 密钥 | **必填** | `sk-...` |
| `LLM_BASE_URL` | API 基础 URL | OpenAI | `https://api.moonshot.cn/v1` |
| `LLM_MODEL` | 模型名称 | `gpt-4o-mini` | `moonshot-v1-8k` |
| `WORKING_DIR` | 工作目录 | 当前目录 | `/home/user/project` |
| `AUTO_CONFIRM` | 自动确认操作 | `false` | `true` / `false` |
| `MAX_TOOL_ITERATIONS` | 最大工具调用次数 | `10` | `5` |

## 使用方法

### 交互模式

启动后进入 REPL 模式，可以直接与 AI 对话：

```bash
$ python -m graph_code

   _____                 _       _____          _
  / ____|               | |     / ____|        | |
 | |  __ _ __ __ _ _ __ | |__  | |     ___   __| | ___
 | | |_ | '__/ _` | '_ \| '_ \ | |    / _ \ / _` |/ _ \
 | |__| | | | (_| | |_) | | | || |___| (_) | (_| |  __/
  \_____|_|  \__,_| .__/|_| |_| \_____\___/ \__,_|\___|
                  | |
                  |_|

[dim]Model: moonshot-v1-8k[/dim]
[dim]Working dir: /Users/gaohong/go/src/graph_code[/dim]

Interactive mode. Type 'exit' or 'quit' to exit.

You: 帮我创建一个计算斐波那契数列的 Python 文件

Graph Code: 我来为您创建这个文件...
[执行工具调用]

已创建文件 fibonacci.py，包含以下内容：
- fibonacci(n) 函数，返回第 n 个斐波那契数
- 示例代码，计算前 10 个斐波那契数

You: exit
Goodbye!
```

### 可用工具

Graph Code 可以调用以下工具：

#### 文件操作
- `read_file` - 读取文件内容，支持行号和范围
- `write_file` - 写入或追加文件内容
- `list_directory` - 列出目录内容
- `glob_search` - 使用 glob 模式搜索文件

#### 代码分析
- `grep_search` - 使用正则表达式搜索代码
- `read_code_chunk` - 读取特定代码片段及其上下文

#### 执行工具
- `bash_command` - 执行 Bash 命令
- `python_execute` - 执行 Python 代码

#### 人机交互
- `ask_user` - 向用户提问获取澄清
- `confirm_action` - 请求确认敏感操作

### 使用示例

**1. 查看项目结构**

```
You: 查看当前目录结构
Graph Code: 当前目录包含以下文件和文件夹：
- .git/ (目录)
- .venv/ (目录)
- graph_code/ (目录)
  - agent/
  - tools/
  - llm/
- README.md (文件, 4KB)
- requirements.txt (文件, 312B)
```

**2. 读取文件**

```
You: 读取 README.md 的前 20 行
Graph Code: 这是 README.md 的前 20 行内容：
[显示内容]
```

**3. 搜索代码**

```
You: 搜索所有 Python 文件中包含 "def " 的行
Graph Code: 找到 15 个匹配项：
- graph_code/main.py:12: def main():
- graph_code/agent/graph.py:25: def build_agent():
...
```

**4. 创建文件**

```
You: 创建一个 hello.py，包含一个 say_hello 函数
Graph Code: 已创建 hello.py：

```python
def say_hello(name: str = "World") -> str:
    """Say hello to someone."""
    return f"Hello, {name}!"

if __name__ == "__main__":
    print(say_hello())
```

**5. 执行命令**

```
You: 运行 hello.py
Graph Code: 执行结果：
Command: python hello.py
Exit code: 0
STDOUT:
Hello, World!
```

**6. 批量操作**

```
You: 找出所有包含 TODO 的文件，并显示相关代码
Graph Code: 搜索中...
找到 3 个文件包含 TODO：
1. graph_code/main.py (第 45 行)
   # TODO: Add error handling
2. graph_code/config.py (第 30 行)
   # TODO: Add validation
```

### 单命令模式

执行单个任务后立即退出：

```bash
# 列出 Python 文件
python -m graph_code "找到所有 Python 文件并统计数量"

# 代码分析
python -m graph_code "分析 graph_code/agent/graph.py 的功能"

# 代码生成
python -m graph_code "创建一个快速排序的实现"
```

### 命令行参数

```bash
python -m graph_code --help

# 主要参数
--api-key TEXT          LLM API key
--base-url TEXT         LLM base URL
--model, -m TEXT        Model name
--working-dir, -w PATH  Working directory
--thread-id, -t TEXT    Thread ID for conversation persistence
--auto-confirm, -y      Auto confirm all actions (use with caution)
--yes                   Answer yes to confirmations (single command mode)
```

## 支持的 LLM

任何兼容 OpenAI API 格式的服务都可以使用：

### 推荐模型

| 提供商 | 推荐模型 | 说明 |
|--------|---------|------|
| Moonshot | `moonshot-v1-8k` | 稳定，支持工具调用 |
| Moonshot | `kimi-k2-turbo-preview` | 性能更好 |
| OpenAI | `gpt-4o-mini` | 性价比高 |
| OpenAI | `gpt-4o` | 功能最强 |

### 配置示例

**Moonshot (Kimi)**
```bash
export LLM_API_KEY=sk-your-key
export LLM_BASE_URL=https://api.moonshot.cn/v1
export LLM_MODEL=moonshot-v1-8k
```

**OpenAI**
```bash
export LLM_API_KEY=sk-your-key
# OpenAI 不需要 BASE_URL，或使用：
# export LLM_BASE_URL=https://api.openai.com/v1
export LLM_MODEL=gpt-4o-mini
```

**Azure OpenAI**
```bash
export LLM_API_KEY=your-azure-key
export LLM_BASE_URL=https://your-resource.openai.azure.com/openai/deployments/your-deployment
export LLM_MODEL=gpt-4
```

**本地模型 (Ollama/vLLM)**
```bash
export LLM_API_KEY=not-needed
export LLM_BASE_URL=http://localhost:11434/v1
export LLM_MODEL=llama3.1
```

## 安全提示

### 操作确认

- 敏感操作（写入文件、执行命令）默认需要确认
- 使用 `--auto-confirm` 或 `AUTO_CONFIRM=true` 可跳过确认（谨慎使用）
- 工具只能访问工作目录内的文件

### 危险命令拦截

以下命令会被自动拦截：
- `rm -rf /` 或 `rm -rf /*`
- 磁盘格式化命令
- Fork 炸弹等

### 工作目录限制

所有文件操作都被限制在工作目录内：
```
WORKING_DIR=/home/user/project

# 允许的操作
read_file("src/main.py")           # ✓
read_file("/home/user/project/src/main.py")  # ✓ 绝对路径但在工作目录内

# 被阻止的操作
read_file("/etc/passwd")           # ✗ 在工作目录外
read_file("../other/file.txt")     # ✗ 尝试跳出工作目录
```

## 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                        Graph Code                           │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────┐  │
│  │  Agent   │──▶│  Tools   │──▶│ Observe  │──▶│ Respond │  │
│  │  Node    │   │  Node    │   │  Node    │   │  Node   │  │
│  └──────────┘   └──────────┘   └──────────┘   └─────────┘  │
│        ▲                                            │       │
│        └────────────────────────────────────────────┘       │
│                      (循环直到完成)                           │
├─────────────────────────────────────────────────────────────┤
│                      LangGraph Engine                       │
└─────────────────────────────────────────────────────────────┘
```

## 故障排除

### 常见问题

**1. API Key 错误**
```
Configuration Error:
  - LLM_API_KEY is required. Set it as environment variable.

解决方案：
export LLM_API_KEY=your-api-key
```

**2. 模型不存在**
```
Error: Not found the model kimi2.5

解决方案：使用正确的模型名，如 moonshot-v1-8k 或 kimi-k2.5
```

**3. 工具调用失败**
```
Error: No matches found for pattern

原因：搜索模式不正确或目录为空
```

**4. 权限错误**
```
Error: Access denied: /etc/passwd is outside working directory

原因：尝试访问工作目录外的文件
解决方案：使用相对路径或在工作目录内操作
```

### 调试模式

启用详细日志：
```bash
export DEBUG=1
python -m graph_code
```

## 开发计划

- [ ] 代码索引和语义搜索
- [ ] 多文件批量编辑
- [ ] MCP (Model Context Protocol) 支持
- [ ] Web UI 界面
- [ ] 更多 LLM 提供商支持
- [ ] kimi-k2.5 推理模型完整支持

## License

MIT License
