"""Explicit synthetic transport for checking the CLI without opening any port."""

from __future__ import annotations

import time


class SimulatedSerial:
    def __init__(self, read_timeout: float, terminator: bytes = b"10", period: float = 0.25):
        self.timeout = read_timeout
        self.terminator = terminator
        self.period = period
        self.is_open = True
        self._buffer = bytearray()
        self._streaming = False
        self._channels: tuple[int, ...] = ()
        self._next_frame = 0.0
        self._counter = 0

    def _produce(self) -> None:
        now = time.monotonic()
        if self._streaming and now >= self._next_frame:
            self._counter += 1
            values = [channel * 100 + self._counter % 100 for channel in self._channels]
            self._buffer.extend(b":100" + "".join(f"{v:04X}" for v in values).encode("ascii") + self.terminator)
            self._next_frame = now + self.period

    @property
    def in_waiting(self) -> int:
        self._produce()
        return len(self._buffer)

    def read(self, size: int) -> bytes:
        deadline = time.monotonic() + self.timeout
        while self.is_open:
            self._produce()
            if self._buffer:
                data = bytes(self._buffer[:size])
                del self._buffer[:size]
                return data
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return b""
            wait = remaining
            if self._streaming:
                wait = min(wait, max(0, self._next_frame - time.monotonic()))
            time.sleep(wait)
        raise OSError("Synthetic port is closed")

    def write(self, data: bytes) -> int:
        if data == b":4\n":
            self._streaming = False
            self._buffer.extend(b":C\n")
        elif data.startswith(b":5") and len(data) == 3:
            self._buffer.extend(b":A\n")
        elif data.startswith(b":2") and len(data) == 99:
            mask = int(data[66:74], 16)
            self._channels = tuple(i + 1 for i in range(32) if mask & (1 << i))
            self._buffer.extend(b":A\n")
        elif data == b":3\n":
            self._streaming = True
            self._next_frame = time.monotonic() + self.period
            self._buffer.extend(b":B\n")
        else:
            raise ValueError(f"Unsupported synthetic command: {data!r}")
        return len(data)

    def reset_input_buffer(self) -> None:
        self._buffer.clear()

    def close(self) -> None:
        self._streaming = False
        self.is_open = False
