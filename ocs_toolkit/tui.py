"""Interactive TUI for building AnswererWrapper configs and managing OCS servers."""

from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional

try:
    import pyperclip
except Exception:  # pragma: no cover - optional dependency fallback
    pyperclip = None

from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app, get_app_or_none
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.filters import Condition
from prompt_toolkit.layout import Dimension, HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.containers import ConditionalContainer, Float, FloatContainer
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import (
    Box,
    Button,
    Checkbox,
    Dialog,
    Frame,
    Label,
    RadioList,
    TextArea,
)
from prompt_toolkit.utils import get_cwidth
class ObservableRadioList(RadioList[str]):
    """RadioList with a simple on-change callback."""

    def __init__(
        self,
        values: List[tuple[str, str]],
        *,
        on_change: Callable[[str], None],
        **kwargs,
    ) -> None:
        super().__init__(values, **kwargs)
        self._on_change = on_change

    def _handle_enter(self) -> None:  # type: ignore[override]
        old = getattr(self, "current_value", None)
        super()._handle_enter()
        if not self.multiple_selection and self.current_value != old:
            self._on_change(self.current_value)

from .server_runtime import start_mock_service, start_ocs_service


@dataclass(frozen=True)
class ProviderPreset:
    key: str
    label: str
    description: str
    base_url: str
    model: str
    requires_key: bool
    header_name: Optional[str]
    header_template: Optional[str]


PRESETS: Dict[str, ProviderPreset] = {
    "copilot": ProviderPreset(
        key="copilot",
        label="Copilot API",
        description="使用 ericc-ch/copilot-api，在 4141 端口提供 OpenAI 兼容接口。",
        base_url="http://localhost:4141/v1",
        model="gpt-5-mini",
        requires_key=False,
        header_name="Authorization",
        header_template="Bearer dummy",
    ),
    "openai": ProviderPreset(
        key="openai",
        label="OpenAI",
        description="OpenAI 官方 API。",
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        requires_key=True,
        header_name="Authorization",
        header_template="Bearer {token}",
    ),
    "ollama": ProviderPreset(
        key="ollama",
        label="Ollama",
        description="本地 Ollama OpenAI 兼容端口。",
        base_url="http://localhost:11434/v1",
        model="llama3.1",
        requires_key=False,
        header_name=None,
        header_template=None,
    ),
    "qwen": ProviderPreset(
        key="qwen",
        label="Qwen",
        description="阿里 DashScope 兼容模式。",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-max",
        requires_key=True,
        header_name="Authorization",
        header_template="Bearer {token}",
    ),
    "custom": ProviderPreset(
        key="custom",
        label="自定义",
        description="任意 OpenAI 兼容服务。",
        base_url="http://localhost:8000/v1",
        model="my-model",
        requires_key=False,
        header_name="Authorization",
        header_template="Bearer {token}",
    ),
}


DEFAULT_OUTPUT_PATH = Path("generated_config.json")
DEFAULT_OCS_HOST = "127.0.0.1"
DEFAULT_OCS_PORT = 8088
DEFAULT_MOCK_HOST = "127.0.0.1"
DEFAULT_MOCK_PORT = 8765
DEFAULT_PLUGIN_URL = "http://127.0.0.1:8088/search"
DEFAULT_PLUGIN_DATA = json.dumps(
    {
        "title": "${title}",
        "question": "${title}",
        "type": "${type}",
        "options": "${options}",
    },
    ensure_ascii=False,
    indent=2,
)
DEFAULT_PLUGIN_HANDLER = (
    "return (res) => {\n"
    "  const matches = Array.isArray(res?.matches)\n"
    "    ? res.matches\n"
    "    : Array.isArray(res?.results)\n"
    "      ? res.results\n"
    "      : undefined;\n"
    "  if (res?.code === 1 && matches?.length) {\n"
    "    return matches;\n"
    "  }\n"
    "  if (Array.isArray(res?.errors) && res.errors.length) {\n"
    "    return res.errors.map(e => [e.message, undefined]);\n"
    "  }\n"
    "  return undefined;\n"
    "};"
)


@dataclass
class ProviderState:
    preset_key: str
    preset_label: str
    base_url: str
    model: str
    temperature: float
    requires_key: bool
    header_name: Optional[str]
    header_template: Optional[str]
    api_token: str


@dataclass
class PluginConfigState:
    name: str
    homepage: str
    url: str
    method: str
    request_type: str
    content_type: str
    data_json: str
    handler_code: str


Stopper = Callable[[], Awaitable[None]]


class InProcessServerManager:
    """Runs an in-process server that returns an async stopper."""

    def __init__(
        self,
        logger: Callable[[str], None],
        on_status: Callable[[str], None],
        label: str,
    ) -> None:
        self._logger = logger
        self._on_status = on_status
        self._label = label
        self._stopper: Optional[Stopper] = None

    async def start(self, starter: Callable[[], Awaitable[Stopper]]) -> None:
        if self._stopper:
            self._logger(f"{self._label} 已在运行，先停止再重启。")
            return
        self._on_status("启动中…")
        try:
            stopper = await starter()
        except Exception:
            self._on_status("启动失败")
            raise
        self._stopper = stopper
        self._on_status("运行中")
        self._logger(f"{self._label} 已启动。")

    async def stop(self) -> None:
        if not self._stopper:
            self._logger(f"当前没有运行中的 {self._label}。")
            return
        stopper = self._stopper
        self._stopper = None
        self._on_status("停止中…")
        try:
            await stopper()
            self._logger(f"{self._label} 已停止。")
        finally:
            self._on_status("已停止")


class TuiApp:
    """Prompt Toolkit based interface for config authoring."""

    def __init__(self) -> None:
        self._status_message = "就绪"
        self._preset_order = list(PRESETS.keys())
        preset_values = [(key, PRESETS[key].label) for key in self._preset_order]
        self.preset_list = ObservableRadioList(
            preset_values,
            select_on_focus=True,
            on_change=self._sync_preset_preview,
        )
        self.base_url_input = self._make_display_field()
        self.model_input = self._make_display_field()
        self.temperature_input = self._make_display_field(text="0.2")
        self.requires_key_checkbox = Checkbox(text="需要 API Key")
        self.header_name_input = self._make_display_field()
        self.header_template_input = self._make_display_field()
        self.api_key_input = self._make_display_field(password=True)
        self.output_path_input = self._make_display_field(text=str(DEFAULT_OUTPUT_PATH))
        self.ocs_host_input = self._make_display_field(text=DEFAULT_OCS_HOST)
        self.ocs_port_input = self._make_display_field(text=str(DEFAULT_OCS_PORT))
        self.mock_host_input = self._make_display_field(text=DEFAULT_MOCK_HOST)
        self.mock_port_input = self._make_display_field(text=str(DEFAULT_MOCK_PORT))
        self.plugin_name_input = self._make_editable_field(text="OCS 题库")
        self.plugin_homepage_input = self._make_editable_field(text="http://localhost")
        self.plugin_url_input = self._make_editable_field(text=DEFAULT_PLUGIN_URL)
        self.plugin_method_input = self._make_editable_field(text="post")
        self.plugin_request_type_input = self._make_editable_field(text="fetch")
        self.plugin_content_type_input = self._make_editable_field(text="json")
        self.plugin_data_input = TextArea(
            text=DEFAULT_PLUGIN_DATA, height=3, wrap_lines=True
        )
        self.plugin_handler_input = TextArea(
            text=DEFAULT_PLUGIN_HANDLER, height=2, wrap_lines=True
        )
        self.preset_description = TextArea(
            height=2, read_only=False, focusable=False, wrap_lines=True
        )
        self.log_view = TextArea(
            text="",
            scrollbar=True,
            wrap_lines=True,
            read_only=False,
            focusable=False,
        )
        self.status_bar = Window(
            height=1,
            content=FormattedTextControl(self._render_status),
            style="class:status",
        )
        self.server_manager = InProcessServerManager(
            self._log, self._set_server_status, "OCS 服务器"
        )
        self.mock_manager = InProcessServerManager(
            self._log, self._set_mock_status, "Mock 服务器"
        )
        self._server_status = "未启动"
        self._mock_status = "未启动"
        self.show_connection_checkbox = Checkbox(text="显示连接配置", checked=False)
        self.show_plugin_checkbox = Checkbox(text="显示插件配置", checked=False)

        self.apply_preset_button = self._make_button(
            text="应用预设", handler=self.handle_apply_preset
        )
        self.edit_connection_button = self._make_button(
            text="编辑连接配置", handler=self.handle_edit_connection
        )
        self.generate_button = self._make_button(
            text="保存配置 ·  F9", handler=self.handle_generate
        )
        self.start_server_button = self._make_button(
            text="启动 OCS ·  F5", handler=self.handle_start_server
        )
        self.stop_server_button = self._make_button(
            text="停止 OCS ·  F6", handler=self.handle_stop_server
        )
        self.copy_plugin_button = self._make_button(
            text="复制插件 ·  F10", handler=self.handle_copy_plugin
        )
        self.start_mock_button = self._make_button(
            text="启动 Mock ·  F7", handler=self.handle_start_mock
        )
        self.stop_mock_button = self._make_button(
            text="停止 Mock ·  F8", handler=self.handle_stop_mock
        )
        self.quit_button = self._make_button(
            text="退出 ·  Ctrl+C", handler=self._exit_app
        )

        root = self._build_layout()
        self._dialog_floats: List[Float] = []
        self._dialog_map: Dict[Dialog, Float] = {}
        container = FloatContainer(content=root, floats=self._dialog_floats)
        kb = self._build_key_bindings()
        self.application = Application(
            layout=Layout(container),
            key_bindings=kb,
            full_screen=True,
            mouse_support=True,
            style=Style.from_dict(
                {
                    "frame.label": "bold",
                    "button.focused": "bg:#00afff #ffffff",
                    "status": "reverse",
                }
            ),
        )

        self.apply_preset(self.preset_list.current_value or self._preset_order[0])

    async def run(self) -> None:
        await self.application.run_async()

    # ------------------------------------------------------------------ UI helpers

    def _build_layout(self) -> HSplit:
        preset_frame = Frame(
            title="模型预设",
            body=HSplit(
                [
                    self.preset_list,
                    Label(text="选中后按 空格/Enter，再点“应用预设”写入字段。"),
                    Frame(title="说明", body=self.preset_description),
                ],
                padding=1,
            ),
        )

        fields = HSplit(
            [
                self._labeled_row("Base URL", self.base_url_input),
                self._labeled_row("Model", self.model_input),
                self._labeled_row("Temperature", self.temperature_input),
                self._labeled_row("Header 名", self.header_name_input),
                self._labeled_row("Header 模板", self.header_template_input),
                self._labeled_row("API Key", self.api_key_input),
                self._labeled_row("输出路径", self.output_path_input),
                self._mini_rows(
                    [
                        ("OCS Host", self.ocs_host_input),
                        ("OCS Port", self.ocs_port_input),
                    ]
                ),
                self._mini_rows(
                    [
                        ("Mock Host", self.mock_host_input),
                        ("Mock Port", self.mock_port_input),
                    ]
                ),
                self.requires_key_checkbox,
            ],
            padding=1,
        )

        toggle_row = VSplit(
            [self.show_connection_checkbox, self.show_plugin_checkbox],
            padding=3,
        )
        preset_row = VSplit(
            [self.apply_preset_button, self.edit_connection_button, self.generate_button],
            padding=2,
        )
        server_row = VSplit(
            [self.start_server_button, self.stop_server_button],
            padding=2,
        )
        mock_row = VSplit(
            [self.copy_plugin_button, self.start_mock_button, self.stop_mock_button],
            padding=2,
        )
        quit_row = VSplit([self.quit_button], padding=0)

        control_bar = Box(
            body=HSplit(
                [
                    toggle_row,
                    preset_row,
                    server_row,
                    mock_row,
                    quit_row,
                ],
                padding=0,
            ),
            padding=0,
            height=Dimension(min=5),
        )

        connection_frame = Frame(title="连接配置", body=fields)
        connection_container = ConditionalContainer(
            content=connection_frame,
            filter=Condition(lambda: self.show_connection_checkbox.checked),
        )

        plugin_frame = Frame(
            title="插件配置",
            body=HSplit(
                [
                    self._labeled_row("名称", self.plugin_name_input),
                    self._labeled_row("主页", self.plugin_homepage_input),
                    self._labeled_row("请求 URL", self.plugin_url_input),
                    self._mini_rows(
                        [
                            ("方法", self.plugin_method_input),
                            ("请求类型", self.plugin_request_type_input),
                            ("内容类型", self.plugin_content_type_input),
                        ]
                    ),
                    Label(text="Data JSON"),
                    self.plugin_data_input,
                    Label(text="Handler JS"),
                    self.plugin_handler_input,
                ],
                padding=1,
            ),
        )

        plugin_container = ConditionalContainer(
            content=plugin_frame,
            filter=Condition(lambda: self.show_plugin_checkbox.checked),
        )

        left = HSplit(
            [
                preset_frame,
                connection_container,
                plugin_container,
                control_bar,
            ],
            padding=1,
        )
        right = Frame(title="运行日志", body=self.log_view)

        return HSplit(
            [
                VSplit([left, right], padding=1),
                Window(height=1, char="─"),
                self.status_bar,
            ]
        )

    def _sync_preset_preview(self, key: Optional[str] = None) -> None:
        preset_key = key or self.preset_list.current_value or self._preset_order[0]
        preset = PRESETS.get(preset_key, PRESETS[self._preset_order[0]])
        self._set_text(self.preset_description, preset.description)

    def _show_dialog(self, dialog: Dialog) -> None:
        float_ref = Float(content=dialog)
        self._dialog_floats.append(float_ref)
        self._dialog_map[dialog] = float_ref
        self.application.layout.focus(dialog)

    def _close_dialog(self, dialog: Dialog) -> None:
        float_ref = self._dialog_map.pop(dialog, None)
        if float_ref:
            with suppress(ValueError):
                self._dialog_floats.remove(float_ref)
        self.application.layout.focus(self.generate_button)

    def _make_display_field(self, text: str = "", password: bool = False) -> TextArea:
        return TextArea(
            text=text,
            height=1,
            password=password,
            multiline=False,
            wrap_lines=False,
            read_only=True,
            focusable=False,
        )

    def _make_editable_field(self, text: str = "", password: bool = False) -> TextArea:
        return TextArea(
            text=text,
            height=1,
            password=password,
            multiline=False,
            wrap_lines=False,
            read_only=False,
            focusable=True,
        )

    def _make_button(self, text: str, handler: Callable[[], None]) -> Button:
        width = max(get_cwidth(text) + 4, 14)
        return Button(
            text=text,
            handler=handler,
            width=width,
            left_symbol="[ ",
            right_symbol=" ]",
        )

    def _make_modal_field(self, text: str = "", password: bool = False) -> TextArea:
        return TextArea(
            text=text,
            height=1,
            password=password,
            multiline=False,
            wrap_lines=False,
        )

    def _labeled_row(self, label: str, widget: TextArea) -> VSplit:
        return VSplit(
            [
                Label(text=f"{label}: ", width=12),
                widget,
            ],
            padding=1,
        )

    def _mini_rows(self, entries: List[tuple[str, TextArea]]) -> HSplit:
        return HSplit(
            [self._labeled_row(label, widget) for label, widget in entries],
            padding=0,
        )

    def _build_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add("c-c")
        @kb.add("c-q")
        def _(event) -> None:
            event.app.exit()

        @kb.add("f9")
        def _(event) -> None:
            self.handle_generate()

        @kb.add("f5")
        def _(event) -> None:
            self.handle_start_server()

        @kb.add("f6")
        def _(event) -> None:
            self.handle_stop_server()

        @kb.add("f7")
        def _(event) -> None:
            self.handle_start_mock()

        @kb.add("f8")
        def _(event) -> None:
            self.handle_stop_mock()

        @kb.add("f10")
        def _(event) -> None:
            self.handle_copy_plugin()

        return kb

    # ------------------------------------------------------------------ Handlers

    def handle_apply_preset(self) -> None:
        key = self.preset_list.current_value or self._preset_order[0]
        self.apply_preset(key)

    def handle_generate(self) -> None:
        try:
            state = self._collect_state()
            payload = build_wrapper_config(state)
            output_path = self._resolve_output_path()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self._log(f"配置已写入 {output_path}")
            self._status_message = f"配置写入 {output_path}"
        except Exception as exc:
            self._log(f"生成配置失败: {exc}")
            self._status_message = "生成失败"
        finally:
            self._refresh_status()

    def handle_start_server(self) -> None:
        async def _runner() -> None:
            try:
                config_path = self._resolve_output_path()
                host = self._get_text(self.ocs_host_input).strip() or DEFAULT_OCS_HOST
                port = self._parse_port(
                    self._get_text(self.ocs_port_input).strip(), DEFAULT_OCS_PORT, "OCS"
                )
                await self.server_manager.start(
                    lambda: start_ocs_service(
                        config_path=config_path,
                        logger=self._log,
                        host=host,
                        port=port,
                    )
                )
            except Exception as exc:
                self._log(f"无法启动 OCS: {exc}")

        get_app().create_background_task(_runner())

    def handle_stop_server(self) -> None:
        async def _runner() -> None:
            try:
                await self.server_manager.stop()
            except Exception as exc:
                self._log(f"停止 OCS 失败: {exc}")

        get_app().create_background_task(_runner())

    def handle_start_mock(self) -> None:
        async def _runner() -> None:
            try:
                host = self._get_text(self.mock_host_input).strip() or DEFAULT_MOCK_HOST
                port = self._parse_port(
                    self._get_text(self.mock_port_input).strip(),
                    DEFAULT_MOCK_PORT,
                    "Mock",
                )
                await self.mock_manager.start(
                    lambda: start_mock_service(
                        logger=self._log,
                        host=host,
                        port=port,
                    )
                )
            except Exception as exc:
                self._log(f"无法启动 Mock: {exc}")

        get_app().create_background_task(_runner())

    def handle_stop_mock(self) -> None:
        async def _runner() -> None:
            try:
                await self.mock_manager.stop()
            except Exception as exc:
                self._log(f"停止 Mock 失败: {exc}")

        get_app().create_background_task(_runner())

    def handle_copy_plugin(self) -> None:
        try:
            state = self._collect_plugin_state()
            payload = build_plugin_config(state)
            json_text = json.dumps(payload, ensure_ascii=False, indent=2)
            copy_to_clipboard(json_text)
            self._log("插件配置已复制到剪贴板，可直接粘贴到浏览器插件中。")
            self._status_message = "插件配置已复制"
        except Exception as exc:
            self._log(f"复制插件配置失败: {exc}")
            self._status_message = "插件复制失败"
        finally:
            self._refresh_status()

    def handle_edit_connection(self) -> None:
        dialog = self._build_connection_dialog()
        self._show_dialog(dialog)

    # ------------------------------------------------------------------ Internal helpers

    def _build_connection_dialog(self) -> Dialog:
        base_input = self._make_modal_field(self._get_text(self.base_url_input))
        model_input = self._make_modal_field(self._get_text(self.model_input))
        temp_input = self._make_modal_field(self._get_text(self.temperature_input))
        header_name_input = self._make_modal_field(self._get_text(self.header_name_input))
        header_tpl_input = self._make_modal_field(self._get_text(self.header_template_input))
        api_key_input = self._make_modal_field(self._get_text(self.api_key_input), password=True)
        output_path_input = self._make_modal_field(self._get_text(self.output_path_input))
        ocs_host_input = self._make_modal_field(self._get_text(self.ocs_host_input))
        ocs_port_input = self._make_modal_field(self._get_text(self.ocs_port_input))
        mock_host_input = self._make_modal_field(self._get_text(self.mock_host_input))
        mock_port_input = self._make_modal_field(self._get_text(self.mock_port_input))
        requires_box = Checkbox(text="需要 API Key", checked=self.requires_key_checkbox.checked)

        dialog_ref: List[Optional[Dialog]] = [None]

        def _save() -> None:
            try:
                base_url = base_input.text.strip()
                model = model_input.text.strip()
                temperature = self._parse_temperature(temp_input.text.strip())
                header_name = header_name_input.text.strip()
                header_tpl = header_tpl_input.text.strip()
                api_key = api_key_input.text.strip()
                output_path = output_path_input.text.strip() or str(DEFAULT_OUTPUT_PATH)
                ocs_host = ocs_host_input.text.strip() or DEFAULT_OCS_HOST
                ocs_port = self._parse_port(ocs_port_input.text.strip(), DEFAULT_OCS_PORT, "OCS")
                mock_host = mock_host_input.text.strip() or DEFAULT_MOCK_HOST
                mock_port = self._parse_port(mock_port_input.text.strip(), DEFAULT_MOCK_PORT, "Mock")
                if not base_url:
                    raise ValueError("Base URL 不能为空。")
                if not model:
                    raise ValueError("Model 不能为空。")
            except Exception as exc:
                self._log(f"更新连接配置失败: {exc}")
                return

            self._set_text(self.base_url_input, base_url)
            self._set_text(self.model_input, model)
            self._set_text(self.temperature_input, str(temperature))
            self._set_text(self.header_name_input, header_name)
            self._set_text(self.header_template_input, header_tpl)
            self._set_text(self.api_key_input, api_key)
            self._set_text(self.output_path_input, output_path)
            self._set_text(self.ocs_host_input, ocs_host)
            self._set_text(self.ocs_port_input, str(ocs_port))
            self._set_text(self.mock_host_input, mock_host)
            self._set_text(self.mock_port_input, str(mock_port))
            self.requires_key_checkbox.checked = requires_box.checked
            if dialog_ref[0] is not None:
                self._close_dialog(dialog_ref[0])

        def _cancel() -> None:
            if dialog_ref[0] is not None:
                self._close_dialog(dialog_ref[0])

        form = HSplit(
            [
                self._labeled_row("Base URL", base_input),
                self._labeled_row("Model", model_input),
                self._labeled_row("Temperature", temp_input),
                self._labeled_row("Header 名", header_name_input),
                self._labeled_row("Header 模板", header_tpl_input),
                self._labeled_row("API Key", api_key_input),
                self._labeled_row("输出路径", output_path_input),
                self._mini_rows(
                    [
                        ("OCS Host", ocs_host_input),
                        ("OCS Port", ocs_port_input),
                    ]
                ),
                self._mini_rows(
                    [
                        ("Mock Host", mock_host_input),
                        ("Mock Port", mock_port_input),
                    ]
                ),
                requires_box,
            ],
            padding=1,
        )

        dialog = Dialog(
            title="编辑连接配置",
            body=form,
            buttons=[
                Button(text="保存", handler=_save),
                Button(text="取消", handler=_cancel),
            ],
            width=80,
        )
        dialog_ref[0] = dialog
        return dialog

    def apply_preset(self, key: str) -> None:
        preset = PRESETS.get(key) or PRESETS[self._preset_order[0]]
        self.preset_list.current_value = preset.key
        self._set_text(self.base_url_input, preset.base_url)
        self._set_text(self.model_input, preset.model)
        self.requires_key_checkbox.checked = preset.requires_key
        self._set_text(self.header_name_input, preset.header_name or "")
        self._set_text(self.header_template_input, preset.header_template or "")
        self._sync_preset_preview(preset.key)
        if preset.requires_key and not self._get_text(self.api_key_input):
            self._set_text(self.api_key_input, "")
        self._status_message = f"已加载预设 {preset.label}"
        self._refresh_status()

    def _collect_state(self) -> ProviderState:
        preset_key = self.preset_list.current_value or self._preset_order[0]
        preset_label = PRESETS.get(preset_key, PRESETS[self._preset_order[0]]).label
        temperature = self._parse_temperature(
            self._get_text(self.temperature_input).strip()
        )
        base_url = self._get_text(self.base_url_input).strip()
        model = self._get_text(self.model_input).strip()
        if not base_url:
            raise ValueError("Base URL 不能为空。")
        if not model:
            raise ValueError("Model 不能为空。")
        return ProviderState(
            preset_key=preset_key,
            preset_label=preset_label,
            base_url=base_url,
            model=model,
            temperature=temperature,
            requires_key=self.requires_key_checkbox.checked,
            header_name=self._get_text(self.header_name_input).strip() or None,
            header_template=self._get_text(self.header_template_input).strip() or None,
            api_token=self._get_text(self.api_key_input).strip(),
        )

    def _collect_plugin_state(self) -> PluginConfigState:
        name = self._get_text(self.plugin_name_input).strip() or "OCS 题库"
        homepage = self._get_text(self.plugin_homepage_input).strip()
        url = self._get_text(self.plugin_url_input).strip()
        method = self._get_text(self.plugin_method_input).strip().lower() or "get"
        request_type = (
            self._get_text(self.plugin_request_type_input).strip()
            or "GM_xmlhttpRequest"
        )
        content_type = self._get_text(self.plugin_content_type_input).strip() or "json"
        data_json = (
            self._get_text(self.plugin_data_input).strip() or DEFAULT_PLUGIN_DATA
        )
        handler_code = (
            self._get_text(self.plugin_handler_input).strip() or DEFAULT_PLUGIN_HANDLER
        )
        if not url:
            raise ValueError("插件请求 URL 不能为空")
        return PluginConfigState(
            name=name,
            homepage=homepage,
            url=url,
            method=method,
            request_type=request_type,
            content_type=content_type,
            data_json=data_json,
            handler_code=handler_code,
        )

    def _resolve_output_path(self) -> Path:
        raw = self._get_text(self.output_path_input).strip() or str(DEFAULT_OUTPUT_PATH)
        return Path(raw).expanduser()

    def _parse_port(self, raw: str, default: int, label: str) -> int:
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"{label} 端口必须是数字。") from exc
        if not 1 <= value <= 65535:
            raise ValueError(f"{label} 端口必须位于 1-65535 之间。")
        return value

    def _parse_temperature(self, raw: str) -> float:
        if not raw:
            return 0.2
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError("Temperature 必须是数字。") from exc
        return max(0.0, min(value, 2.0))

    def _set_text(self, widget: TextArea, value: str) -> None:
        buffer = widget.buffer
        buffer.set_document(
            Document(text=value, cursor_position=len(value)),
            bypass_readonly=True,
        )

    def _get_text(self, widget: TextArea) -> str:
        return widget.buffer.text

    def _log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}\n"
        buffer = self.log_view.buffer
        buffer.insert_text(line, move_cursor=True)
        buffer.cursor_position = len(buffer.text)

    def _render_status(self) -> FormattedText:
        return FormattedText(
            [
                (
                    "class:status",
                    f" 状态: {self._status_message} | OCS: {self._server_status} | Mock: {self._mock_status} ",
                ),
            ]
        )

    def _refresh_status(self) -> None:
        app = get_app_or_none()
        if app:
            app.invalidate()

    def _set_server_status(self, text: str) -> None:
        self._server_status = text
        self._refresh_status()

    def _set_mock_status(self, text: str) -> None:
        self._mock_status = text
        self._refresh_status()

    def _exit_app(self) -> None:
        async def _shutdown() -> None:
            await self.server_manager.stop()
            await self.mock_manager.stop()
            self.application.exit()

        get_app().create_background_task(_shutdown())


def build_wrapper_config(state: ProviderState) -> List[Dict[str, object]]:
    """Convert ProviderState into an AnswererWrapper-compatible config."""

    headers: Dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if state.header_name and state.header_template:
        token = state.api_token.strip()
        if "{token}" in state.header_template and not token and state.requires_key:
            raise ValueError("当前预设需要 API Key，但输入为空。")
        rendered = state.header_template
        if "{token}" in state.header_template:
            rendered = state.header_template.format(token=token)
        headers[state.header_name] = rendered

    endpoint = f"{state.base_url.rstrip('/')}/chat/completions"
    display_name = f"{state.preset_label} ({state.model})"
    data = {
        "model": state.model,
        "messages": {"handler": "build_gpt5_messages"},
        "temperature": state.temperature,
    }
    return [
        {
            "name": display_name,
            "url": endpoint,
            "method": "post",
            "contentType": "json",
            "headers": headers,
            "data": data,
            "handler": "gpt5_response_handler",
        }
    ]


def build_plugin_config(state: PluginConfigState) -> List[Dict[str, object]]:
    """Create a browser-plugin-friendly config block."""

    try:
        data_obj = json.loads(state.data_json)
    except json.JSONDecodeError as exc:
        raise ValueError("插件 Data JSON 不是合法 JSON") from exc
    if not isinstance(data_obj, dict):
        raise ValueError("插件 Data JSON 顶层必须是对象")
    handler_code = state.handler_code.strip() or DEFAULT_PLUGIN_HANDLER
    return [
        {
            "name": state.name,
            "homepage": state.homepage or None,
            "url": state.url,
            "method": state.method,
            "type": state.request_type,
            "contentType": state.content_type,
            "data": data_obj,
            "handler": handler_code,
        }
    ]


def copy_to_clipboard(text: str) -> None:
    if pyperclip is None:
        raise RuntimeError("当前环境未安装 pyperclip，无法复制到剪贴板。")
    try:
        pyperclip.copy(text)
    except Exception as exc:  # pragma: no cover - platform specific
        raise RuntimeError(f"复制到剪贴板失败: {exc}") from exc


async def _run_async() -> None:
    app = TuiApp()
    await app.run()


def main() -> None:
    asyncio.run(_run_async())


if __name__ == "__main__":
    main()
