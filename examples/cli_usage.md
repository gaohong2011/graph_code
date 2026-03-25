# CLI 使用示例

Graph Code 支持多种命令行使用方式。

## 1. 交互模式

启动交互式对话：

```bash
python -m graph_code
```

在交互模式下：
- 输入 `help` 或 `?` 显示帮助
- 输入 `exit`、`quit` 或 `q` 退出
- 使用上下箭头浏览历史

## 2. 单命令模式

执行单个命令后退出：

```bash
python -m graph_code "列出当前目录的所有 Python 文件"
python -m graph_code "读取 README.md 的前 20 行"
python -m graph_code "搜索所有包含 'TODO' 的文件"
```

## 3. 指定工作目录

在不同目录下运行：

```bash
python -m graph_code -w /path/to/project
python -m graph_code --working-dir /path/to/project "分析这个项目的结构"
```

## 4. 使用不同的模型

临时切换模型：

```bash
# 使用 Moonshot 模型
python -m graph_code --model moonshot-v1-8k

# 使用 OpenAI 模型
python -m graph_code --model gpt-4o-mini
```

## 5. 自动确认模式

跳过所有确认提示（谨慎使用）：

```bash
# 交互模式自动确认
python -m graph_code --auto-confirm

# 单命令模式自动确认
python -m graph_code --yes "删除所有 .pyc 文件"
```

## 6. 完整的命令行参数

```bash
python -m graph_code \
  --api-key sk-your-key \
  --base-url https://api.moonshot.cn/v1 \
  --model moonshot-v1-8k \
  --working-dir /path/to/project \
  --auto-confirm \
  "列出所有 Python 文件并统计数量"
```

## 7. 组合使用示例

### 批量处理文件

```bash
# 查找并分析代码
python -m graph_code "
    1. 找到所有包含 'class ' 的 Python 文件
    2. 读取第一个类定义所在的文件
    3. 分析这个类的功能
"

# 生成文档
python -m graph_code "
    1. 列出 graph_code/tools/ 目录的所有文件
    2. 读取每个文件的内容
    3. 为每个模块生成文档
"
```

### 代码审查

```bash
# 检查代码风格
python -m graph_code -w ./my-project "
    检查项目中的代码问题：
    - 找出所有行数超过 100 行的函数
    - 找出所有未使用的导入
    - 检查是否缺少文档字符串
"

# 安全检查
python -m graph_code -w ./my-project "
    检查潜在的安全问题：
    - 搜索所有 eval() 的使用
    - 搜索所有 subprocess 调用
    - 检查是否有硬编码的密码或密钥
"
```

### 项目初始化

```bash
# 在新项目中创建基础结构
mkdir new-project
cd new-project

python -m graph_code "
    创建一个 Python 项目的基础结构：
    1. 创建 src/ 目录和 __init__.py
    2. 创建 tests/ 目录
    3. 创建 README.md 模板
    4. 创建 requirements.txt
    5. 创建 .gitignore
"
```

## 8. 环境变量配置

推荐通过环境变量配置，避免在命令行暴露密钥：

```bash
# ~/.bashrc 或 ~/.zshrc
export LLM_API_KEY="sk-your-key"
export LLM_BASE_URL="https://api.moonshot.cn/v1"
export LLM_MODEL="moonshot-v1-8k"

# 可选配置
export WORKING_DIR="/home/user/projects"
export AUTO_CONFIRM="false"
```

然后可以直接运行：

```bash
python -m graph_code
```

## 9. 别名配置

在 shell 中创建快捷方式：

```bash
# ~/.bashrc 或 ~/.zshrc
alias gc='python -m graph_code'
alias gcp='python -m graph_code --working-dir $(pwd)'

# 使用
$ gc "列出所有文件"
$ gcp "分析当前项目"
```

## 10. 管道和重定向

结合其他命令使用：

```bash
# 将结果保存到文件
python -m graph_code "分析项目结构" > analysis.txt

# 使用 echo 传递多行输入
echo "
1. 列出所有 Python 文件
2. 找出最大的文件
3. 读取这个文件的内容
" | python -m graph_code

# 结合 find 命令
find . -name "*.py" -type f | head -10 | xargs -I {} python -m graph_code "分析 {}"
```

## 11. 调试模式

启用详细输出进行调试：

```bash
export DEBUG=1
python -m graph_code "测试命令"
```

## 12. 多会话管理

使用不同的 thread-id 保持独立的会话：

```bash
# 项目 A 的会话
python -m graph_code -t project-a -w ./project-a

# 项目 B 的会话
python -m graph_code -t project-b -w ./project-b
```
