# Graph Code 示例

这个目录包含 Graph Code 的使用示例。

## 快速示例

### 1. 基本交互 (basic_interaction.md)

展示基本的文件操作和代码分析。

```bash
python -m graph_code
```

**示例对话：**
```
You: 列出当前目录的所有 Python 文件
You: 读取 graph_code/main.py 的内容
You: 搜索所有包含 "def " 的函数定义
```

### 2. 代码生成 (code_generation.md)

展示如何生成和修改代码。

```bash
python -m graph_code "创建一个计算斐波那契数列的 Python 脚本"
```

### 3. 代码重构 (code_refactoring.md)

展示如何分析和重构现有代码。

```bash
python -m graph_code "重构这个函数，添加类型注解"
```

## 示例脚本

### hello_world.py

简单的示例文件，可以被 Graph Code 读取和修改。

```bash
python -m graph_code "读取 examples/hello_world.py 并添加一个新的函数"
```

## 使用场景

### 场景 1: 探索新项目

```bash
# 进入项目目录
cd my-project

# 启动 Graph Code
export LLM_API_KEY=your-key
export LLM_BASE_URL=https://api.moonshot.cn/v1
python -m graph_code

# 对话示例
You: 这是什么项目？分析目录结构和主要文件
You: 找出所有的 Python 类定义
You: 这个项目的主要入口点在哪里？
```

### 场景 2: 添加新功能

```bash
# 直接执行命令
python -m graph_code "创建一个 User 类，包含 name 和 email 属性"

python -m graph_code "为 User 类添加一个 validate_email 方法"
```

### 场景 3: 代码审查

```bash
python -m graph_code "检查所有 Python 文件的代码风格问题"

python -m graph_code "找出所有未使用的导入"
```

### 场景 4: 自动化任务

```bash
# 批量重命名文件
python -m graph_code "将所有 test_*.py 重命名为 *_test.py"

# 生成文档
python -m graph_code "为所有函数生成 docstring"
```

## 提示词技巧

### 清晰的指令

✓ **好的提示词：**
```
读取 src/utils.py，找到 process_data 函数，
添加错误处理，当输入为 None 时返回空列表
```

✗ **不好的提示词：**
```
改一下 utils.py
```

### 使用文件路径

✓ **好的提示词：**
```
分析 graph_code/agent/graph.py 中的 build_agent 函数
```

✗ **不好的提示词：**
```
分析 agent 的代码
```

### 分步骤任务

✓ **好的提示词：**
```
第一步：列出所有测试文件
第二步：读取 test_main.py 的内容
第三步：为这个测试文件添加更多的测试用例
```

### 明确期望的输出

✓ **好的提示词：**
```
创建一个 Python 函数，接收一个列表，返回排序后的列表。
使用快速排序算法，添加类型注解和文档字符串。
```

## 更多资源

- [主文档](../README.md)
- [配置示例](../.env.example)
- [API 文档](https://platform.moonshot.cn/docs) (Moonshot)
