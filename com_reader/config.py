"""Validated settings; channel numbers in JSON are physical, one-based IDs."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Literal, overload


class ConfigError(ValueError):
    pass


def _object(value: Any, name: str, allowed: set[str]) -> dict:
    if not isinstance(value, dict):
        raise ConfigError(f"{name}: ожидается JSON-объект")
    unknown = set(value) - allowed
    if unknown:
        raise ConfigError(f"{name}: неизвестные поля: {', '.join(sorted(unknown))}")
    return value


def _integer(value: Any, name: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ConfigError(f"{name}: ожидается целое число {low}…{high}")
    return value


@overload
def _boolean(value: Any, name: str, nullable: Literal[False] = False) -> bool: ...


@overload
def _boolean(value: Any, name: str, nullable: Literal[True]) -> bool | None: ...


def _boolean(value: Any, name: str, nullable: bool = False) -> bool | None:
    if value is None and nullable:
        return None
    if type(value) is not bool:
        raise ConfigError(f"{name}: ожидается true/false" + (" или null" if nullable else ""))
    return value


def _timeout(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ConfigError(f"{name}: ожидается положительное конечное число секунд")
    return float(value)


@dataclass(frozen=True)
class SerialSettings:
    port: str
    baudrate: int = 128000
    rtscts: bool = False
    dsrdtr: bool = False
    xonxoff: bool = False
    rts: bool | None = None
    dtr: bool | None = None


@dataclass(frozen=True)
class MeasurementSettings:
    channels: tuple[int, ...]
    channel_delays: tuple[int, ...]
    interval_seconds: int
    filter_hz: int
    mode: str = "single"

    @property
    def minimum_cycle_seconds(self) -> int:
        # MainFrm.cpp stores StrProfile::GetMinTime() in an int before sending it.
        return sum(self.channel_delays[channel - 1] for channel in self.channels) // 25


@dataclass(frozen=True)
class ProtocolSettings:
    data_terminators: tuple[bytes, ...] = (b"10", b"\r\n", b"\n")
    read_timeout_seconds: float = 0.1
    command_timeout_seconds: float = 2.0
    sample_timeout_seconds: float = 30.0


@dataclass(frozen=True)
class Config:
    serial: SerialSettings
    measurement: MeasurementSettings
    protocol: ProtocolSettings
    legacy: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Any) -> Config:
        data = _object(data, "config", {"schema_version", "serial", "measurement", "protocol", "legacy"})
        _integer(data.get("schema_version"), "schema_version", 1, 1)
        s = _object(data.get("serial"), "serial", {"port", "baudrate", "rtscts", "dsrdtr", "xonxoff", "rts", "dtr"})
        port = s.get("port")
        if not isinstance(port, str) or not port.strip():
            raise ConfigError("serial.port: укажите COM-порт")
        serial = SerialSettings(
            port=port.strip(),
            baudrate=_integer(s.get("baudrate", 128000), "serial.baudrate", 1, 4_000_000),
            rtscts=_boolean(s.get("rtscts", False), "serial.rtscts"),
            dsrdtr=_boolean(s.get("dsrdtr", False), "serial.dsrdtr"),
            xonxoff=_boolean(s.get("xonxoff", False), "serial.xonxoff"),
            rts=_boolean(s.get("rts"), "serial.rts", nullable=True),
            dtr=_boolean(s.get("dtr"), "serial.dtr", nullable=True),
        )
        m = _object(data.get("measurement"), "measurement", {"mode", "channels", "channel_delays", "interval_seconds", "filter_hz"})
        if m.get("mode") != "single":
            raise ConfigError("Поддерживается только measurement.mode = 'single'; Cross не реализован")
        channels = m.get("channels")
        if not isinstance(channels, list) or not channels:
            raise ConfigError("measurement.channels: нужен непустой список физических каналов 1…32")
        channels = tuple(_integer(c, "measurement.channels", 1, 32) for c in channels)
        if tuple(sorted(set(channels))) != channels:
            raise ConfigError("measurement.channels: номера должны быть уникальными и отсортированными")
        delays = m.get("channel_delays")
        if not isinstance(delays, list) or len(delays) != 32:
            raise ConfigError("measurement.channel_delays: нужны 32 значения для физических каналов")
        delays = tuple(_integer(d, "measurement.channel_delays", 0, 256) for d in delays)
        if any(delays[c - 1] == 0 for c in channels):
            raise ConfigError("Задержка включённого канала должна быть в диапазоне 1…256")
        interval = _integer(m.get("interval_seconds"), "measurement.interval_seconds", 1, 6553)
        frequency = _integer(m.get("filter_hz"), "measurement.filter_hz", 50, 500)
        if frequency not in (50, 250, 500):
            raise ConfigError("measurement.filter_hz: допустимы только 50, 250, 500")
        measurement = MeasurementSettings(channels, delays, interval, frequency)
        if interval < measurement.minimum_cycle_seconds:
            raise ConfigError("Интервал меньше оценки длительности цикла из старого кода")
        p = _object(data.get("protocol", {}), "protocol", {"data_terminators_hex", "read_timeout_seconds", "command_timeout_seconds", "sample_timeout_seconds"})
        hex_terms = p.get("data_terminators_hex", ["3130", "0d0a", "0a"])
        if not isinstance(hex_terms, list) or not hex_terms:
            raise ConfigError("protocol.data_terminators_hex: нужен непустой список")
        terms = []
        for item in hex_terms:
            if not isinstance(item, str):
                raise ConfigError("Окончания пакетов должны быть hex-строками")
            try:
                term = bytes.fromhex(item)
            except ValueError as exc:
                raise ConfigError(f"Некорректное окончание пакета: {item!r}") from exc
            if not 1 <= len(term) <= 8 or term in terms:
                raise ConfigError("Окончания пакетов должны быть уникальными, длиной 1…8 байт")
            terms.append(term)
        if any(a.startswith(b) for a in terms for b in terms if a != b):
            raise ConfigError("Одно окончание пакета не должно быть префиксом другого")
        protocol = ProtocolSettings(
            data_terminators=tuple(terms),
            read_timeout_seconds=_timeout(p.get("read_timeout_seconds", 0.1), "read_timeout_seconds"),
            command_timeout_seconds=_timeout(p.get("command_timeout_seconds", 2.0), "command_timeout_seconds"),
            sample_timeout_seconds=_timeout(p.get("sample_timeout_seconds", 30.0), "sample_timeout_seconds"),
        )
        if protocol.read_timeout_seconds > min(protocol.command_timeout_seconds, protocol.sample_timeout_seconds):
            raise ConfigError("Таймаут одного чтения не должен превышать таймаут команды/измерения")
        legacy = data.get("legacy", {})
        if not isinstance(legacy, dict):
            raise ConfigError("legacy: ожидается JSON-объект")
        if legacy.get("zero_offsets_applied", False) is not False:
            raise ConfigError("Применение поправок нуля не реализовано; zero_offsets_applied должен быть false")
        return cls(serial, measurement, protocol, legacy)

    @classmethod
    def load(cls, path: Path | str) -> Config:
        try:
            return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8-sig")))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ConfigError(f"Не удалось прочитать настройки {path}: {exc}") from exc
