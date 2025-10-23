from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Dict, Iterator, Optional

import paramiko


@dataclass
class SSHCredentials:
    host: str
    port: int
    username: str
    password: str


class RemoteSession:
    def __init__(self) -> None:
        self._client: Optional[paramiko.SSHClient] = None

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
