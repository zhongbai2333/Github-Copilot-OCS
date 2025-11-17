"""Adapter utilities for AnswererWrapper-compatible APIs."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from urllib import parse, request

HandlerReturn = Union[
    None,
    Sequence[Union[str, None]],
    Sequence[Sequence[Union[str, None]]],
]

def flatten_openai_content(content: Any) -> str:
    """Flatten OpenAI-style message content into a plain string."""

    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                if isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif block.get("type") == "image_url":
                    parts.append("[image]")
                elif isinstance(block.get("content"), str):
                    parts.append(block["content"])
                else:
                    parts.append(json.dumps(block, ensure_ascii=False))
            else:
                parts.append(str(block))
        return "\n".join(part for part in parts if part.strip()).strip()
    return str(content).strip()


def _stringify_error(value: Any) -> str:
    """Serialize an error payload into human readable text."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _first_choice_content(payload: Dict[str, Any]) -> Optional[str]:
    """Extract the first textual message content from an OpenAI response."""
    choices = payload.get("choices")
    if not isinstance(choices, Sequence) or not choices:
        return None
    first = choices[0]
    if isinstance(first, dict):
        message = first.get("message") or {}
        content = message.get("content")
        if content is None and isinstance(first.get("delta"), dict):
            content = first["delta"].get("content")
        text = flatten_openai_content(content)
        return text or None
    return None


def build_gpt5_messages(env: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Construct a GPT-5 mini prompt from the current question context."""

    title = str(env.get("title", "") or "").strip()
    options_list = build_labeled_options(env)
    is_multi = _is_multi_select(env)

    prompt_lines: List[str] = []
    if title:
        prompt_lines.append(f"Question: {title}")
    options_materialized = list(options_list)
    if options_materialized:
        prompt_lines.append("Options (respond using the letter labels only):")
        prompt_lines.extend(options_materialized)
    if is_multi:
        prompt_lines.append(
            "This is a multiple-answer question. Return only the letters for the options that are truly correct; do not include every option."
        )
    else:
        prompt_lines.append(
            "This is a single-answer question. Return only the one best letter; never include extra letters."
        )
    prompt_lines.append(
        "Provide concise reasoning and end with 'ANSWER: <letters>' such as 'ANSWER: A' or 'ANSWER: A,B'."
    )

    user_prompt = "\n".join(prompt_lines).strip()
    return [
        {
            "role": "system",
            "content": "You are GPT-5 mini assisting with exam questions. Reply in Simplified Chinese unless instructed otherwise.",
        },
        {"role": "user", "content": user_prompt},
    ]


def gpt5_response_handler(res: Dict[str, Any]) -> HandlerReturn:
    """Convert an OpenAI-style completion into AnswererWrapper output."""

    answer = _first_choice_content(res)
    if answer:
        return [[None, answer]]
    error_message = _stringify_error(res.get("error") or res.get("message") or "No answer returned")
    return [[error_message, None]]


SAFE_EVAL_GLOBALS: Dict[str, Any] = {
    "__builtins__": {
        "len": len,
        "min": min,
        "max": max,
        "sum": sum,
        "sorted": sorted,
        "any": any,
        "all": all,
        "map": map,
        "filter": filter,
    },
    "math": math,
    "re": re,
    "json": json,
    "flatten_openai_content": flatten_openai_content,
    "build_gpt5_messages": build_gpt5_messages,
    "gpt5_response_handler": gpt5_response_handler,
}


OPTION_PREFIX_PATTERN = re.compile(r"^\s*([A-Z0-9]+)[\.．、\)]\s*(.+)$", re.IGNORECASE)
MULTI_SELECT_KEYWORDS = (
    "multi",
    "multiple",
    "multi-select",
    "multi choice",
    "multi-choice",
    "多选",
    "复选",
    "多项",
)


@dataclass(frozen=True)
class AnswerMatch:
    """Single question-answer tuple emitted by a handler."""

    question: Optional[str]
    answer: Optional[str]
    source: str


@dataclass(frozen=True)
class AdapterError:
    """Execution error captured for a single wrapper config."""

    source: str
    message: str


@dataclass
class AdapterResult:
    """Collection of matches and errors for a single adapter run."""

    matches: List[AnswerMatch] = field(default_factory=list)
    errors: List[AdapterError] = field(default_factory=list)


@dataclass
class AnswererWrapperConfig:
    """Serializable representation of a single wrapper config."""

    name: str
    url: str
    handler: Callable[[Any], HandlerReturn]
    homepage: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    method: str = "GET"
    content_type: str = "json"
    request_type: str = "fetch"
    headers: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: Dict[str, Any]) -> "AnswererWrapperConfig":
        _require_keys(mapping, ["name", "url", "handler"])
        handler_callable = _compile_callable(mapping["handler"], f"handler({mapping['name']})")
        method = mapping.get("method", mapping.get("Method", "get")).upper()
        content_type = _coalesce(mapping, ["contentType", "content_type"], default="json").lower()
        request_type = mapping.get("type", mapping.get("requestType", "fetch"))
        headers = mapping.get("headers", {}) or {}
        return cls(
            name=mapping["name"],
            url=mapping["url"],
            handler=handler_callable,
            homepage=mapping.get("homepage"),
            data=mapping.get("data", {}) or {},
            method=method,
            content_type=content_type,
            request_type=request_type,
            headers=headers,
        )


class HttpRequester:
    """Thin HTTP client based on urllib."""

    def __init__(self, opener: Optional[Callable[..., Any]] = None, timeout: float = 15.0) -> None:
        self._opener = opener or request.urlopen
        self._timeout = timeout

    def fetch(self, config: AnswererWrapperConfig, payload: Dict[str, Any]) -> bytes:
        method = config.method.upper()
        headers = dict(config.headers)
        if method == "GET":
            target = _build_query_url(config.url, payload)
            req = request.Request(url=target, headers=headers, method="GET")
            data = None
        else:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers.setdefault("Content-Type", "application/json; charset=utf-8")
            data = body
            req = request.Request(url=config.url, data=data, headers=headers, method="POST")
        with self._opener(req, timeout=self._timeout) as resp:
            return resp.read()


class AnswererWrapperAdapter:
    """Executes one or more AnswererWrapper configs in Python."""

    def __init__(
        self,
        configs: Sequence[AnswererWrapperConfig],
        requester: Optional[HttpRequester] = None,
    ) -> None:
        self.configs = list(configs)
        self._requester = requester or HttpRequester()

    def run(self, env: Dict[str, Any]) -> AdapterResult:
        context = _normalize_env(env)
        result = AdapterResult()
        for config in self.configs:
            try:
                payload = _resolve_data_tree(config.data, context)
                raw = self._requester.fetch(config, payload)
                parsed = _parse_body(raw, config.content_type)
                matches = _normalize_handler_output(config.handler(parsed), config.name)
                result.matches.extend(matches)
            except Exception as exc:
                result.errors.append(AdapterError(source=config.name, message=str(exc)))
        return result


def load_config_file(path: Union[str, Path]) -> List[AnswererWrapperConfig]:
    """Load wrapper configs from a JSON file."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Config root must be a list")
    return [AnswererWrapperConfig.from_mapping(item) for item in data]


# ---------------------------------------------------------------------------
# Helpers


def _coalesce(mapping: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _require_keys(mapping: Dict[str, Any], keys: Sequence[str]) -> None:
    missing = [k for k in keys if k not in mapping]
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")


def _compile_callable(value: Any, label: str) -> Callable[[Any], Any]:
    if callable(value):
        return value
    if not isinstance(value, str):
        raise TypeError(f"{label} must be callable or python expression string")
    compiled = eval(value, SAFE_EVAL_GLOBALS, {})
    if not callable(compiled):
        raise TypeError(f"{label} did not evaluate to a callable")
    return compiled


def _normalize_env(env: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {"title": "", "type": "", "options": ""}
    for key, value in env.items():
        if value is not None:
            normalized[key] = value
    options = normalized.get("options", "")
    if isinstance(options, str):
        normalized.setdefault("options_list", [line for line in options.splitlines() if line.strip()])
    return normalized


def build_labeled_options(env: Dict[str, Any]) -> List[str]:
    raw_options = env.get("options_list") or env.get("options") or []
    if isinstance(raw_options, str):
        option_values = [line.strip() for line in raw_options.splitlines() if line.strip()]
    elif isinstance(raw_options, Iterable):
        option_values = [str(item).strip() for item in raw_options if str(item).strip()]
    else:
        option_values = []

    labeled: List[str] = []
    for index, option in enumerate(option_values):
        match = OPTION_PREFIX_PATTERN.match(option)
        if match:
            label = match.group(1).upper()
            text = match.group(2).strip()
            labeled.append(f"{label}. {text}")
            continue
        label = _generate_label(index)
        labeled.append(f"{label}. {option}")
    return labeled


def _generate_label(index: int) -> str:
    if index < 26:
        return chr(ord("A") + index)
    return f"Option{index + 1}"


def _is_multi_select(env: Dict[str, Any]) -> bool:
    qtype = str(env.get("type", "") or "").lower()
    if qtype and any(keyword in qtype for keyword in MULTI_SELECT_KEYWORDS):
        return True
    question = str(env.get("title") or env.get("question") or "").lower()
    if question and any(keyword in question for keyword in MULTI_SELECT_KEYWORDS):
        return True
    return False


def _resolve_data_tree(tree: Any, env: Dict[str, Any]) -> Any:
    if isinstance(tree, dict):
        if set(tree.keys()) == {"handler"}:
            func = _compile_callable(tree["handler"], "data.handler")
            return func(env)
        return {key: _resolve_data_tree(value, env) for key, value in tree.items()}
    if isinstance(tree, list):
        return [_resolve_data_tree(item, env) for item in tree]
    if isinstance(tree, str):
        return _apply_placeholders(tree, env)
    return tree


def _apply_placeholders(value: str, env: Dict[str, Any]) -> str:
    str_env = {key: str(val) for key, val in env.items() if isinstance(val, (str, int, float))}
    return Template(value).safe_substitute(str_env)


def _build_query_url(base: str, params: Dict[str, Any]) -> str:
    if not params:
        return base
    serialized = {
        key: _stringify_query_value(value)
        for key, value in params.items()
    }
    query = parse.urlencode(serialized, doseq=True)
    separator = "&" if "?" in base else "?"
    return f"{base}{separator}{query}" if query else base


def _stringify_query_value(value: Any) -> str:
    if isinstance(value, (str, int, float)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def _parse_body(raw: bytes, content_type: str) -> Any:
    text = raw.decode("utf-8")
    if content_type == "json":
        return json.loads(text)
    return text


def _normalize_handler_output(raw: HandlerReturn, source: str) -> List[AnswerMatch]:
    matches: List[AnswerMatch] = []
    if raw is None:
        return matches
    if _looks_like_pair(raw):
        question, answer = raw  # type: ignore[index]
        matches.append(AnswerMatch(question=question, answer=answer, source=source))
        return matches
    if not isinstance(raw, Sequence):
        raise TypeError("Handler must return a pair or sequence of pairs")
    for item in raw:
        if not _looks_like_pair(item):
            raise TypeError("Handler returned an invalid entry; expected [question, answer]")
        question, answer = item  # type: ignore[index]
        matches.append(AnswerMatch(question=question, answer=answer, source=source))
    return matches


def _looks_like_pair(candidate: Any) -> bool:
    return isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)) and len(candidate) == 2
