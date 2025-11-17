# Github Copilot OCS

Helper resources for experimenting with OCS/AnswererWrapper style 题库 APIs.

## Repository layout

- `ocs_toolkit/` – Consolidated adapter library, in-process servers, CLI, and prompt-toolkit TUI.
- `python_adapter/` – Sample JSON configs kept for reference (no executable code).
- `tests/` – Unit tests that validate the adapter logic.

## Python adapter quickstart

1. (Optional) run the mock service to emulate a题库 endpoint:

```cmd
python -m ocs_toolkit.mock_server
```

1. Execute the CLI against the sample config:

```cmd
python -m ocs_toolkit.cli --config python_adapter\example_config.json --title "1+2" --question-type single --options "A.1\nB.2" --pretty
```

The CLI prints JSON that contains both successful matches and any per-config errors.

## TUI 配置器

运行交互式 TUI，可以快速为不同的 OpenAI 兼容模型生成配置（Copilot API/OpenAI/Qwen/Ollama/自定义）：

```cmd
python main.py
```

- 预设多个常用供应商（Copilot/OpenAI/Ollama/Qwen/自定义），按 `空格` 选中后点击 `应用预设` 即可自动填充 Base URL、模型和鉴权头。
- 可额外自定义 Base URL/模型 ID/Temperature/Header 模板/API Key/输出路径，默认写入 `ocs_toolkit/generated_config.json`。
- `F9` 保存配置、`F5/F6` 启动/停止内置 OCS 服务（直接在当前进程加载生成的配置并监听指定 Host/Port），`F7/F8` 一键启动/停止仓库自带的 Mock 服务器，日志面板实时滚动并根据终端宽高自适应。
- 填写“插件配置”区块后按 `F10`（或“复制插件”按钮）即可将符合浏览器脚本插件格式的题库配置 JSON 写入剪贴板，直接粘贴进插件列表。
- “插件配置”默认折叠，可勾选 `显示插件配置` 复选框再填写字段，降低小终端使用门槛。
- 生成的配置可直接传给 `python -m ocs_toolkit.cli --config <生成文件>` 进行答题测试。

## Testing

Run the adapter unit tests whenever you touch the Python sources:

```cmd
python -m unittest
```
