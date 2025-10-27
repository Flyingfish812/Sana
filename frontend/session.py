from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Dict, Iterator, Optional

import paramiko
import requests
from urllib.parse import urlunparse
import json as _json
import requests

@dataclass
class SSHCredentials:
    host: str
    port: int
    username: str
    password: str


class RemoteSession:
    def __init__(self) -> None:
        self._client: Optional[paramiko.SSHClient] = None
        self._base_url: Optional[str] = None
        self._remote_home: Optional[str] = None

    def connect(self, creds: SSHCredentials, *, timeout: float = 10.0) -> None:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            creds.host,
            port=creds.port,
            username=creds.username,
            password=creds.password,
            timeout=timeout,
        )
        self._client = client
        host = creds.host.strip()
        port = 8000
        self._base_url = f"http://{host}:{port}"
        try:
            out = self.exec_simple('bash -lc \'printf %s "$HOME"\'')
            self._remote_home = out.strip() or None
        except Exception:
            self._remote_home = None
    
    def _expand_remote_path(self, path: str) -> str:
        """将以 ~ 开头的路径展开为真实远端 HOME 路径。"""
        if not path:
            return path
        if path.startswith("~"):
            # 若 connect() 时拿到了 HOME，就做本地替换；否则退回原样
            if self._remote_home:
                if path == "~":
                    return self._remote_home
                # "~" + remainder
                return self._remote_home + path[1:]
        return path


    def is_active(self) -> bool:
        if self._client is None:
            return False
        transport = self._client.get_transport()
        return bool(transport and transport.is_active())

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def stream_jsonl(self, command: str) -> Iterator[Dict[str, object]]:
        if not self.is_active():
            raise RuntimeError("SSH session is not connected.")

        assert self._client is not None
        transport = self._client.get_transport()
        if transport is None:
            raise RuntimeError("SSH transport is not available.")
        channel = transport.open_session()
        channel.exec_command(command)
        stdout = channel.makefile("r")
        stderr = channel.makefile_stderr("r")
        error_text = ""
        exit_status = 0

        try:
            while True:
                line = stdout.readline()
                if not line:
                    if channel.exit_status_ready():
                        break
                    time.sleep(0.1)
                    continue
                text = line.strip()
                if not text:
                    continue
                try:
                    event = json.loads(text)
                except json.JSONDecodeError:
                    event = {"type": "log", "level": "INFO", "logger": "remote", "message": text}
                yield event
        finally:
            error_text = stderr.read().strip()
            exit_status = channel.recv_exit_status()
            stdout.close()
            stderr.close()
            channel.close()

        if error_text:
            yield {"type": "log", "level": "ERROR", "logger": "stderr", "message": error_text}
        if exit_status != 0:
            yield {"type": "error", "message": f"Remote command exited with status {exit_status}"}
    
    def exec_simple(self, command: str, *, timeout: float | None = None) -> str:
        """执行简单命令并返回 stdout 文本（一次性）。"""
        if not self.is_active():
            raise RuntimeError("SSH session is not connected.")
        assert self._client is not None
        stdin, stdout, stderr = self._client.exec_command(command, timeout=timeout)
        out = stdout.read().decode("utf-8", errors="ignore")
        err = stderr.read().decode("utf-8", errors="ignore")
        rc = stdout.channel.recv_exit_status()
        if rc != 0:
            raise RuntimeError(f"Command failed ({rc}): {err.strip() or out.strip()}")
        return out

    def list_yaml_configs(self, directory: str) -> list[str]:
        """列出目录下的 .yaml / .yml 文件（仅文件名，不含路径）。"""
        # 兼容 zsh/bash，使用 -1 保证一行一个
        cmd = f'bash -lc "ls -1 {directory}/*.y*ml 2>/dev/null || true"'
        out = self.exec_simple(cmd)
        files = [line.strip() for line in out.splitlines() if line.strip()]
        # 仅返回文件名
        return [f.split("/")[-1] for f in files]

    def read_text_file(self, path: str) -> str:
        """用 SFTP 读取远端文本文件。"""
        if not self.is_active():
            raise RuntimeError("SSH session is not connected.")
        path = self._expand_remote_path(path)
        assert self._client is not None
        sftp = self._client.open_sftp()
        try:
            with sftp.open(path, "r") as fh:
                return fh.read().decode("utf-8", errors="ignore")
        finally:
            sftp.close()
    
    def get_base_url(self) -> str:
        if not self._base_url:
            raise RuntimeError("HTTP base URL is not configured. Did you call connect()?")
        return self._base_url

    def http_stream_ndjson(self, path: str, json_body: dict, *, timeout: float = 5.0):
        """
        以 NDJSON 流方式 POST 并逐行 yield 事件 dict。
        """
        base = self.get_base_url()
        url = base + path  # 例如 "/train/run"
        try:
            with requests.post(url, json=json_body, stream=True, timeout=timeout) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        # 兼容后端偶发的纯文本日志行
                        yield {"type": "log", "message": line}
        except requests.RequestException as exc:
            yield {"type": "error", "message": f"HTTP error: {exc}"}
    
    def http_post_json(self, path: str, json_body: dict, *, timeout: float = 30.0) -> dict:
        """POST JSON，返回 JSON（用于 /data/run /viz/one-click 等非流式接口）。"""
        base = self.get_base_url()
        url = base + path
        resp = requests.post(url, json=json_body, timeout=timeout)
        resp.raise_for_status()
        try:
            return resp.json()
        except Exception:
            # 兜底：可能返回文本
            return {"ok": True, "raw": resp.text}
