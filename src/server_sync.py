"""
Helpers for sending parking updates to a remote server.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


@dataclass
class ServerSyncClient:
    endpoint: str | None = None
    api_key: str | None = None
    timeout_seconds: float = 5.0
    min_interval_seconds: float = 0.0
    _last_sent_at: float = field(default=0.0, init=False)
    _last_payload: str | None = field(default=None, init=False)

    def send_update(self, payload: dict) -> bool:
        if not self.endpoint:
            return False

        now = time.time()
        encoded_payload = json.dumps(payload, sort_keys=True)
        if encoded_payload == self._last_payload:
            return False

        if self.min_interval_seconds and now - self._last_sent_at < self.min_interval_seconds:
            return False

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = urllib.request.Request(
            self.endpoint,
            data=encoded_payload.encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                if 200 <= response.status < 300:
                    self._last_sent_at = now
                    self._last_payload = encoded_payload
                    return True
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            print(f"[WARN] Could not send server update: {error}")

        return False
