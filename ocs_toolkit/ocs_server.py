"""HTTP server that exposes AnswererWrapper configs via a local endpoint."""

from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlparse

from .answerer_wrapper import AnswerMatch, AnswererWrapperAdapter, load_config_file

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8088
Logger = Callable[[str], None]


def create_ocs_server(
    config_path: Path,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    logger: Logger | None = None,
) -> HTTPServer:
    """Create an HTTP server backed by the provided AnswererWrapper config."""

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    configs = load_config_file(config_path)
    if not configs:
        raise ValueError("配置文件为空，无法启动 OCS 服务器。")

    adapter = AnswererWrapperAdapter(configs)
    log = logger or (lambda message: None)

    class AdapterHandler(BaseHTTPRequestHandler):
        def do_HEAD(self) -> None:  # noqa: N802
            if self._normalized_path() not in {"/", "/search"}:
                self._write_empty(404)
                return
            self._write_empty()

        def do_OPTIONS(self) -> None:  # noqa: N802
            if self._normalized_path() not in {"/", "/search"}:
                self._write_empty(404)
                return
            self.send_response(204)
            self._write_cors_headers()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            if self._normalized_path() not in {"/", "/search"}:
                self._write_json(404, {"code": 0, "message": "Unknown path"})
                return
            query = parse_qs(urlparse(self.path).query, keep_blank_values=True)
            payload = {key: values[-1] for key, values in query.items()}
            self._run_adapter(payload)

        def do_POST(self) -> None:  # noqa: N802
            if self._normalized_path() not in {"/", "/search"}:
                self._write_json(404, {"code": 0, "message": "Unknown path"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                self._write_json(400, {"code": 0, "message": "Invalid JSON"})
                return
            self._run_adapter(payload)

        def log_message(self, format: str, *args: Tuple[object, ...]) -> None:  # noqa: A003
            log(format % args)

        def _run_adapter(self, payload: Dict[str, object]) -> None:
            env = _build_env(payload)
            log(f"OCS 请求: {env['title']}")
            if env.get("options_list"):
                options_preview = " | ".join(env["options_list"][:6])
                log(f"选项: {options_preview}")
            result = adapter.run(env)
            context_title = env.get("title") or "(空题干)"
            matches = _format_matches(result.matches, env)
            response = {
                "code": 1 if matches else 0,
                "matches": matches,
                "results": matches,
                "raw_answers": [
                    [match.question or context_title, match.answer or ""]
                    for match in result.matches
                ],
                "errors": [error.__dict__ for error in result.errors],
                "msg": "success" if matches else "no matches",
            }
            log(
                f"OCS 结果: {len(result.matches)} matches, {len(result.errors)} errors"
            )
            if result.matches:
                preview = " | ".join(
                    f"{match.question or '(空题干)'} -> {match.answer or '(空答案)'}"
                    for match in result.matches[:3]
                )
                log(f"AI 返回示例: {preview}")
            elif result.errors:
                error_preview = "; ".join(
                    f"{error.source}: {error.message}" for error in result.errors[:3]
                )
                log(f"AI 报错: {error_preview}")
            self._write_json(200, response)

        def _write_json(self, status: int, payload: Dict[str, object]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self._write_cors_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _write_empty(self, status: int = 200) -> None:
            self.send_response(status)
            self._write_cors_headers()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _normalized_path(self) -> str:
            return urlparse(self.path).path or "/"

        def _write_cors_headers(self) -> None:
            origin = self.headers.get("Origin") or "*"
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header(
                "Access-Control-Allow-Methods", "GET,POST,HEAD,OPTIONS"
            )
            self.send_header("Access-Control-Allow-Headers", "*, Content-Type")
            self.send_header("Access-Control-Allow-Credentials", "true")

    return HTTPServer((host, port), AdapterHandler)


def _build_env(payload: Dict[str, object]) -> Dict[str, object]:
    title = str(payload.get("title") or payload.get("question") or "").strip()
    qtype = str(payload.get("type") or payload.get("question_type") or "single")
    raw_options = payload.get("options") or payload.get("options_list") or ""
    if isinstance(raw_options, list):
        options_list = [str(item).strip() for item in raw_options if str(item).strip()]
        serialized = "\n".join(options_list)
    else:
        serialized = str(raw_options)
        options_list = [line.strip() for line in serialized.splitlines() if line.strip()]
    return {
        "title": title,
        "type": qtype,
        "options": serialized,
        "options_list": options_list,
    }


OPTION_PATTERN = re.compile(r"^\s*([A-Z0-9]+)[\.．、\)]\s*(.+)$", re.IGNORECASE)
ANSWER_PATTERN = re.compile(r"(?:answer|答案)\s*[:：]\s*([A-Z0-9 ,、，.;；]+)", re.IGNORECASE)
TRUE_FALSE_TYPE_KEYWORDS = (
    "judge",
    "truefalse",
    "true-false",
    "true_false",
    "tf",
    "判断",
    "对错",
    "判断题",
)
TRUE_TOKENS = (
    "对",
    "正确",
    "是",
    "yes",
    "true",
    "√",
    "right",
)
FALSE_TOKENS = (
    "错",
    "错误",
    "否",
    "no",
    "false",
    "×",
    "wrong",
)


def _format_matches(matches: Sequence[AnswererWrapperAdapter.AnswerMatch], env: Dict[str, object]) -> List[List[str]]:  # type: ignore[attr-defined]
    options = _parse_option_entries(env.get("options_list") or [])
    true_false_map = _build_true_false_map(env, options)
    context_title = env.get("title") or "(空题干)"
    formatted: List[List[str]] = []
    for match in matches:
        question_text = match.question or context_title
        answer_text = (match.answer or "").strip()
        mapped_options = _map_answer_to_option(answer_text, options, true_false_map)
        if mapped_options:
            for option_text in mapped_options:
                formatted.append([question_text, option_text])
        else:
            formatted.append([question_text, answer_text])
    return formatted


def _parse_option_entries(options_list: Sequence[str]) -> List[Tuple[Optional[str], str]]:
    entries: List[Tuple[Optional[str], str]] = []
    for index, option in enumerate(options_list):
        text = option.strip()
        match = OPTION_PATTERN.match(text)
        if match:
            label = match.group(1).upper()
            entries.append((label, text))
        else:
            label = _generate_label(index)
            normalized = text
            if label:
                normalized = f"{label}. {text}"
            entries.append((label, normalized))
    return entries


def _map_answer_to_option(
    answer_text: str,
    options: Sequence[Tuple[Optional[str], str]],
    true_false_map: Optional[Dict[str, str]] = None,
) -> Optional[List[str]]:
    if not answer_text:
        return None
    resolved: List[str] = []

    def _append_option(label: Optional[str], text: str) -> None:
        display: Optional[str]
        if true_false_map:
            display = None
            if label and label in true_false_map:
                display = true_false_map[label]
            else:
                stripped = _strip_option_label(text)
                display = true_false_map.get(stripped) or true_false_map.get(text)
            if not display:
                display = label or stripped or text
        else:
            display = label or text
        if display and display not in resolved:
            resolved.append(display)

    marker = ANSWER_PATTERN.search(answer_text)
    if marker:
        for label in _split_label_tokens(marker.group(1)):
            candidate = _find_option_by_label(label, options)
            if candidate:
                _append_option(*candidate)
        if resolved:
            return resolved

    normalized = answer_text.upper()
    for label, full_text in options:
        if label and label in normalized:
            _append_option(label, full_text)
            continue
        if full_text and full_text in answer_text:
            _append_option(label, full_text)

    if resolved:
        return resolved
    return None


def _find_option_by_label(label: str, options: Sequence[Tuple[Optional[str], str]]) -> Optional[Tuple[Optional[str], str]]:
    for option_label, text in options:
        if option_label == label:
            return option_label, text
    return None


def _build_true_false_map(
    env: Dict[str, object], options: Sequence[Tuple[Optional[str], str]]
) -> Dict[str, str]:
    qtype = str(env.get("type") or "").lower()
    is_judge = any(keyword in qtype for keyword in TRUE_FALSE_TYPE_KEYWORDS)
    if not is_judge and not _looks_like_true_false_options(options):
        return {}
    mapping: Dict[str, str] = {}
    for label, text in options:
        normalized = _normalize_true_false_text(text)
        if not normalized:
            continue
        if label:
            mapping[label] = normalized
        stripped = _strip_option_label(text)
        mapping[stripped] = normalized
        mapping[text] = normalized
    if len(set(mapping.values())) < 2:
        return {}
    return mapping


def _looks_like_true_false_options(options: Sequence[Tuple[Optional[str], str]]) -> bool:
    if not 1 <= len(options) <= 3:
        return False
    hits = 0
    for _, text in options:
        if _normalize_true_false_text(text):
            hits += 1
    return hits >= 2


def _normalize_true_false_text(text: str) -> Optional[str]:
    stripped = _strip_option_label(text)
    lower = stripped.lower()
    if any(token in stripped for token in TRUE_TOKENS) or any(
        token in lower for token in ("yes", "true", "right")
    ):
        return "对"
    if any(token in stripped for token in FALSE_TOKENS) or any(
        token in lower for token in ("no", "false", "wrong")
    ):
        return "错"
    return None


def _strip_option_label(text: str) -> str:
    match = OPTION_PATTERN.match(text.strip())
    if match:
        return match.group(2).strip()
    parts = text.strip().split(maxsplit=1)
    if len(parts) == 2 and len(parts[0]) == 1 and parts[0].isalpha():
        return parts[1].strip()
    return text.strip()


def _split_label_tokens(block: str) -> List[str]:
    tokens: List[str] = []
    raw_parts = re.split(r"[^A-Z0-9]+", block.upper())
    for part in raw_parts:
        if not part:
            continue
        if len(part) > 1 and part.isalpha():
            tokens.extend(list(part))
        else:
            tokens.append(part)
    return tokens


def _generate_label(index: int) -> Optional[str]:
    if index < 26:
        return chr(ord("A") + index)
    if index < 52:
        return "A" + chr(ord("A") + (index - 26))
    return str(index + 1)