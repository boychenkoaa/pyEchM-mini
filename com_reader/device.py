"""Single-owner serial worker with a software-triggered, one-sample API."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time
from typing import Protocol, TextIO

from .config import Config, SerialSettings
from .protocol import FrameParser, START_COMMAND, STOP_COMMAND, configure_command, filter_command


class DeviceError(RuntimeError):
    pass


class CommandError(DeviceError):
    pass


class SampleTimeout(DeviceError):
    pass


@dataclass(frozen=True)
class Sample:
    timestamp: datetime
    sequence: int
    channels: tuple[int, ...]
    values: tuple[int, ...]
    reference_id: int


class SerialTransport(Protocol):
    @property
    def in_waiting(self) -> int: ...
    def read(self, size: int, /) -> bytes: ...
    def write(self, data: bytes, /) -> int | None: ...
    def reset_input_buffer(self) -> None: ...
    def close(self) -> None: ...


SerialFactory = Callable[[SerialSettings, float, float], SerialTransport]


def open_serial(settings: SerialSettings, read_timeout: float, write_timeout: float) -> SerialTransport:
    try:
        import serial
    except ImportError as exc:
        raise DeviceError("Не установлен pyserial. Выполните python -m pip install -r requirements.txt в .venv") from exc
    port = serial.Serial(
        port=None,
        baudrate=settings.baudrate,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=read_timeout,
        write_timeout=write_timeout,
        rtscts=settings.rtscts,
        dsrdtr=settings.dsrdtr,
        xonxoff=settings.xonxoff,
    )
    try:
        if settings.rts is not None:
            port.rts = settings.rts
        if settings.dtr is not None:
            port.dtr = settings.dtr
        port.port = settings.port
        port.open()
    except BaseException:
        port.close()
        raise
    return port


class Device:
    def __init__(self, config: Config, *, log: Callable[[str], None] | None = None,
                 trace_path: Path | str | None = None, serial_factory: SerialFactory | None = None):
        self.config = config
        self._log = log or (lambda message: None)
        self._trace_path = Path(trace_path) if trace_path is not None else None
        self._serial_factory = serial_factory or open_serial
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._started = threading.Event()
        self._thread: threading.Thread | None = None
        self._ready = False
        self._error: BaseException | None = None
        self._shutdown_error: BaseException | None = None
        self._sequence = 0
        self._request_ns: int | None = None
        self._requested_sample: Sample | None = None
        self._port: SerialTransport | None = None
        self._trace_file: TextIO | None = None
        self._trace_error: Exception | None = None
        self._last_read_ns = 0
        self.parser = FrameParser(len(config.measurement.channels), config.protocol.data_terminators)

    @property
    def is_running(self) -> bool:
        """Whether the serial owner may still be accessing the port."""
        return self._thread is not None and self._thread.is_alive()

    def _trace(self, direction: str, data: bytes) -> None:
        if self._trace_file is not None and self._trace_error is None:
            record = {"timestamp": datetime.now(timezone.utc).isoformat(), "direction": direction,
                      "hex": data.hex(), "length": len(data)}
            try:
                self._trace_file.write(json.dumps(record) + "\n")
                self._trace_file.flush()
            except (OSError, ValueError) as exc:
                self._trace_error = exc
                raise DeviceError(f"Ошибка записи трассировки: {exc}") from exc

    def _read(self) -> bytes:
        assert self._port is not None
        data = self._port.read(max(1, min(self._port.in_waiting, 4096)))
        self._last_read_ns = time.monotonic_ns()
        if data:
            self._trace("rx", data)
        return data

    def _write(self, command: bytes, *, trace_optional: bool = False) -> None:
        assert self._port is not None
        try:
            self._trace("tx", command)
        except DeviceError:
            if not trace_optional:
                raise
            self._log("Трассировка недоступна; всё равно отправляю аварийный STOP.")
        written = self._port.write(command)
        if written != len(command):
            raise CommandError(f"Неполная запись команды: {written}/{len(command)} байт")

    def _exchange(self, command: bytes, expected: bytes | None, name: str, *, allow_stopped: bool = False) -> bytes:
        # Purge only at command boundaries, never for a software sample trigger.
        if self._stop.is_set() and not allow_stopped:
            raise CommandError(f"{name}: обмен отменён")
        assert self._port is not None
        self._port.reset_input_buffer()
        deadline = time.monotonic() + self.config.protocol.command_timeout_seconds
        self._write(command, trace_optional=allow_stopped)
        response = bytearray()
        while time.monotonic() < deadline:
            if self._stop.is_set() and not allow_stopped:
                raise CommandError(f"{name}: обмен отменён")
            response.extend(self._read())
            if expected is not None:
                marker = b":" + expected
                if marker in response:
                    return bytes(response)
            else:
                # Legacy SetFiltrFreq checks only that some response was received.
                # Do not invent an ACK code: retain/log it for hardware verification.
                start = response.find(b":")
                while 0 <= start < len(response) - 1:
                    if response[start + 1] != ord("1"):
                        return bytes(response)
                    start = response.find(b":", start + 1)
            if len(response) > 65536:
                del response[:-4096]
        detail = bytes(response[-64:]).hex() or "нет данных"
        raise CommandError(f"{name}: нет подтверждения за {self.config.protocol.command_timeout_seconds:g} с; RX: {detail}")

    def start(self) -> None:
        if self._thread is not None:
            raise DeviceError("Этот экземпляр Device уже запускался")
        self._thread = threading.Thread(target=self._run, name="com-reader", daemon=True)
        self._thread.start()
        budget = self.config.protocol.command_timeout_seconds * 6 + self.config.protocol.read_timeout_seconds * 6 + 5
        if not self._started.wait(budget):
            self.close()
            raise DeviceError("Поток подключения не завершил инициализацию вовремя")
        with self._condition:
            if self._error is not None:
                raise DeviceError(f"Не удалось запустить приём: {self._error}") from self._error
            if not self._ready:
                raise DeviceError("Приём не запущен")

    def wait_next_sample(self, timeout: float | None = None) -> Sample:
        """Wait for the first full frame whose first host-read byte follows this call.

        This is not a hardware trigger: data can predate the call inside the device
        or OS serial buffers. A partially received frame is explicitly excluded.
        """
        if timeout is None:
            timeout = self.config.protocol.sample_timeout_seconds
        if timeout <= 0:
            raise ValueError("Sample timeout must be positive")
        deadline = time.monotonic() + timeout
        with self._condition:
            if not self._ready or self._stop.is_set():
                raise DeviceError("Приём не запущен или уже остановлен")
            if self._request_ns is not None:
                raise DeviceError("Другой запрос измерения уже ожидает результат")
            self._request_ns = time.monotonic_ns()
            self._requested_sample = None
            try:
                while self._requested_sample is None:
                    if self._error is not None:
                        raise DeviceError(f"Ошибка приёма: {self._error}") from self._error
                    if self._stop.is_set():
                        raise DeviceError("Приём остановлен")
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise SampleTimeout(
                            f"За {timeout:g} с не получен новый полный набор. Проверьте прибор, "
                            "число каналов и окончания пакетов; при необходимости включите --trace. "
                            f"Некорректных пакетов: {self.parser.invalid_frames}; "
                            f"отброшено байт: {self.parser.discarded_bytes}."
                        )
                    self._condition.wait(remaining)
                return self._requested_sample
            finally:
                self._request_ns = None
                self._requested_sample = None

    def _publish(self, frame) -> None:
        with self._condition:
            self._sequence += 1
            if (self._request_ns is not None and self._requested_sample is None
                    and frame.first_received_ns >= self._request_ns):
                self._requested_sample = Sample(
                    timestamp=datetime.now(timezone.utc), sequence=self._sequence,
                    channels=self.config.measurement.channels, values=frame.values,
                    reference_id=frame.reference_id,
                )
                self._condition.notify_all()

    def _run(self) -> None:
        start_attempted = False
        try:
            if self._trace_path is not None:
                self._trace_file = self._trace_path.open("a", encoding="utf-8")
            self._port = self._serial_factory(
                self.config.serial, self.config.protocol.read_timeout_seconds,
                self.config.protocol.command_timeout_seconds,
            )
            self._log(f"Порт {self.config.serial.port} открыт: {self.config.serial.baudrate} бод, 8N1.")
            for attempt in range(3):
                if self._stop.is_set():
                    raise DeviceError("Подключение отменено")
                try:
                    self._exchange(STOP_COMMAND, b"C", "Остановка перед настройкой")
                    break
                except CommandError:
                    if attempt == 2:
                        raise
                    self._log("Прибор не подтвердил остановку; повторяю команду.")
            reply = self._exchange(filter_command(self.config.measurement.filter_hz), None, "Настройка фильтра")
            self._log(f"Команда фильтра {self.config.measurement.filter_hz} Гц: ответ {reply.hex()}. "
                      "Код подтверждения фильтра не установлен по старым исходникам.")
            self._exchange(configure_command(self.config.measurement), b"A", "Настройка измерений")
            # Keep data returned in the same read as the START acknowledgment.
            start_attempted = True
            response = self._exchange(START_COMMAND, b"B", "Запуск измерений")
            with self._condition:
                self._ready = True
            self._started.set()
            seen_terminators = set()
            while not self._stop.is_set():
                chunk = response if response else self._read()
                response = b""
                received_ns = self._last_read_ns
                for frame in self.parser.feed(chunk, received_ns):
                    if frame.terminator not in seen_terminators:
                        seen_terminators.add(frame.terminator)
                        self._log(f"Обнаружено окончание пакета: {frame.terminator.hex()}.")
                    self._publish(frame)
        except Exception as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()
            self._log(f"Ошибка фонового приёма: {exc}")
        finally:
            self._started.set()
            if self._port is not None:
                try:
                    if start_attempted:
                        self._exchange(STOP_COMMAND, b"C", "Остановка при завершении", allow_stopped=True)
                        self._log("Прибор подтвердил остановку.")
                except Exception as exc:
                    self._shutdown_error = exc
                    self._log(f"Не удалось подтвердить остановку прибора: {exc}")
                finally:
                    try:
                        self._port.close()
                    except Exception as exc:
                        self._shutdown_error = exc
            if self._trace_file is not None:
                try:
                    self._trace_file.close()
                except (OSError, ValueError) as exc:
                    self._trace_error = exc
            with self._condition:
                self._ready = False
                self._condition.notify_all()
            self._log(f"Приём завершён. Пакетов: {self._sequence}; "
                      f"отброшено байт: {self.parser.discarded_bytes}; "
                      f"некорректных пакетов: {self.parser.invalid_frames}.")

    def close(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            # Allow an interrupted command/write plus the complete final STOP.
            self._thread.join(2 * self.config.protocol.command_timeout_seconds + 2 * self.config.protocol.read_timeout_seconds + 2)
            if self._thread.is_alive():
                raise DeviceError("Поток COM-порта не завершился; остановка прибора не подтверждена")
        if self._shutdown_error is not None:
            raise DeviceError(f"Порт закрыт, но корректное завершение не подтверждено: {self._shutdown_error}")
        if self._error is not None:
            raise DeviceError(f"Приём завершился с ошибкой: {self._error}") from self._error
        if self._trace_error is not None:
            raise DeviceError(f"Ошибка записи трассировки: {self._trace_error}") from self._trace_error

    def __enter__(self) -> Device:
        try:
            self.start()
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
