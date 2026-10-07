"""Wire commands and an incremental parser, independent of serial I/O and UI."""

from __future__ import annotations

from dataclasses import dataclass
import re
import time

from .config import MeasurementSettings

START_COMMAND = b":3\n"
STOP_COMMAND = b":4\n"
_HEX = re.compile(rb"[0-9a-fA-F]+")


def filter_command(frequency_hz: int) -> bytes:
    try:
        code = {50: 1, 250: 2, 500: 3}[frequency_hz]
    except KeyError as exc:
        raise ValueError("Filter frequency must be 50, 250 or 500 Hz") from exc
    return b":5" + bytes([code])


def configure_command(settings: MeasurementSettings) -> bytes:
    """Reproduce MIPSSetAdcChar for Single, including integer timing truncation."""
    if settings.mode != "single":
        raise ValueError("Only Single mode is implemented")
    if len(settings.channel_delays) != 32:
        raise ValueError("Exactly 32 channel delays are required")
    if not settings.channels or tuple(sorted(set(settings.channels))) != settings.channels:
        raise ValueError("Channels must be nonempty, sorted and unique")
    if any(type(c) is not int or not 1 <= c <= 32 for c in settings.channels):
        raise ValueError("Physical channel IDs must be 1..32")
    if any(type(d) is not int or not 0 <= d <= 256 for d in settings.channel_delays):
        raise ValueError("Channel delays must be 0..256")
    if any(settings.channel_delays[c - 1] < 1 for c in settings.channels):
        raise ValueError("Enabled channel delays must be positive")
    mask = sum(1 << (c - 1) for c in settings.channels)
    delay_fields = "".join(f"{max(0, delay - 1):02X}" for delay in settings.channel_delays)
    pause = max(1, int((settings.interval_seconds - settings.minimum_cycle_seconds) * 10))
    if pause > 0xFFFF:
        raise ValueError("Encoded measurement pause exceeds four hex digits")
    packet = f":2{delay_fields}{mask:08X}00000000{pause:04X}{'0' * 12}\n"
    return packet.encode("ascii")


def signed_word(field: bytes) -> int:
    if len(field) != 4 or not _HEX.fullmatch(field):
        raise ValueError("A value must contain exactly four ASCII hex characters")
    value = int(field, 16)
    return value - 0x10000 if value >= 0x8000 else value


@dataclass(frozen=True)
class Frame:
    reference_id: int
    values: tuple[int, ...]
    raw: bytes
    first_received_ns: int
    completed_ns: int
    terminator: bytes


class FrameParser:
    """Recognize only complete frames; preserve fragments and resynchronize on ':1'.

    Arrival timestamps are host read timestamps, NOT instrument acquisition times.
    Tracking the first byte lets callers exclude a frame already partially read
    before a software trigger. The bytes may still have waited in a driver buffer.
    """

    def __init__(self, channel_count: int, terminators: tuple[bytes, ...]):
        if type(channel_count) is not int or not 1 <= channel_count <= 32:
            raise ValueError("Channel count must be 1..32")
        if not terminators or any(not term for term in terminators):
            raise ValueError("At least one nonempty terminator is required")
        self.channel_count = channel_count
        self.terminators = terminators
        self._payload_length = 4 + channel_count * 4
        self._buffer = bytearray()
        self._arrival_times: list[int] = []
        self.discarded_bytes = 0
        self.invalid_frames = 0

    @property
    def buffered_bytes(self) -> int:
        return len(self._buffer)

    def _drop(self, count: int, *, discarded: bool = False) -> None:
        del self._buffer[:count]
        del self._arrival_times[:count]
        if discarded:
            self.discarded_bytes += count

    def feed(self, data: bytes, received_ns: int | None = None) -> list[Frame]:
        if received_ns is None:
            received_ns = time.monotonic_ns()
        self._buffer.extend(data)
        self._arrival_times.extend([received_ns] * len(data))
        frames = []
        while self._buffer:
            start = self._buffer.find(b":1")
            if start < 0:
                keep = 1 if self._buffer.endswith(b":") else 0
                self._drop(len(self._buffer) - keep, discarded=True)
                break
            if start:
                self._drop(start, discarded=True)
            if len(self._buffer) < self._payload_length:
                # A new header inside a truncated frame cannot be valid hex data.
                next_start = self._buffer.find(b":1", 2)
                if next_start >= 0:
                    self.invalid_frames += 1
                    self._drop(next_start, discarded=True)
                    continue
                break
            fields = bytes(self._buffer[2:self._payload_length])
            if not _HEX.fullmatch(fields) or int(fields[:2], 16) >= 32:
                self.invalid_frames += 1
                self._drop(1, discarded=True)
                continue
            rest = bytes(self._buffer[self._payload_length:])
            terminator = next((term for term in self.terminators if rest.startswith(term)), None)
            if terminator is None:
                if not rest or any(term.startswith(rest) for term in self.terminators):
                    break
                self.invalid_frames += 1
                self._drop(1, discarded=True)
                continue
            length = self._payload_length + len(terminator)
            values = tuple(signed_word(fields[i:i + 4]) for i in range(2, len(fields), 4))
            frames.append(Frame(
                reference_id=int(fields[:2], 16),
                values=values,
                raw=bytes(self._buffer[:length]),
                first_received_ns=self._arrival_times[0],
                completed_ns=self._arrival_times[length - 1],
                terminator=terminator,
            ))
            self._drop(length)
        return frames
