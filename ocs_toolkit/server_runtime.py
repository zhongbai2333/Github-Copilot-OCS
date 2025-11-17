"""Utilities for running bundled HTTP servers inside the TUI process."""

from __future__ import annotations

import asyncio
import threading
from http.server import HTTPServer
from pathlib import Path
from typing import Awaitable, Callable

from .mock_server import DEFAULT_HOST as MOCK_HOST, DEFAULT_PORT as MOCK_PORT, create_mock_server
from .ocs_server import (
    DEFAULT_HOST as OCS_HOST,
    DEFAULT_PORT as OCS_PORT,
    create_ocs_server,
)

Logger = Callable[[str], None]
Stopper = Callable[[], Awaitable[None]]


async def start_mock_service(
    logger: Logger,
    *,
    host: str = MOCK_HOST,
    port: int = MOCK_PORT,
) -> Stopper:
    server = create_mock_server(host=host, port=port, logger=logger)
    return await _start_http_server(server, f"Mock 服务器 ({host}:{port})", logger)


async def start_ocs_service(
    config_path: Path,
    logger: Logger,
    *,
    host: str = OCS_HOST,
    port: int = OCS_PORT,
) -> Stopper:
    server = create_ocs_server(config_path=config_path, host=host, port=port, logger=logger)
    return await _start_http_server(server, f"OCS 服务器 ({host}:{port})", logger)


async def _start_http_server(server: HTTPServer, label: str, logger: Logger) -> Stopper:
    loop = asyncio.get_running_loop()
    stopped = loop.create_future()

    def _serve() -> None:
        addr = _format_address(server)
        logger(f"{label} 启动，监听 http://{addr}")
        try:
            server.serve_forever()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger(f"{label} 异常退出: {exc}")
            raise
        finally:
            loop.call_soon_threadsafe(stopped.set_result, None)

    thread = threading.Thread(target=_serve, name=label, daemon=True)
    thread.start()

    async def _stop() -> None:
        logger(f"正在停止 {label} …")
        server.shutdown()
        await stopped
        server.server_close()
        thread.join(timeout=1)
        logger(f"{label} 已停止。")

    return _stop


def _format_address(server: HTTPServer) -> str:
    host, port = server.server_address[:2]
    return f"{host}:{port}"
