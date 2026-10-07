"""Human prompts go to stderr; stdout contains only CSV records."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
import sys
from typing import Protocol

from .config import Config, ConfigError
from .device import Device, DeviceError, Sample, SampleTimeout
from .protocol import configure_command
from .simulation import SimulatedSerial

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.json"
HELP = "Enter — получить следующий набор; h + Enter — справка; q + Enter или Ctrl+C — выход."


def _report(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def describe(config: Config) -> None:
    _report(f"Порт: {config.serial.port}; {config.serial.baudrate} бод, 8N1.")
    _report("Режим: Single. Физические каналы / порядок CSV: " + ", ".join(map(str, config.measurement.channels)))
    _report(f"Интервал из профиля: {config.measurement.interval_seconds} с; фильтр: {config.measurement.filter_hz} Гц.")
    _report("Значения прибора: signed int16; физические единицы не установлены; поправки нуля не применяются.")
    if config.legacy.get("original_mode") == "cross":
        _report("ВНИМАНИЕ: исходный профиль был Cross. Сейчас выбран Single; это другой режим измерения.")
    if config.serial.rts is None or config.serial.dtr is None:
        _report("RTS/DTR = null: используются значения pyserial по умолчанию. Эквивалентность старому порту не проверена.")
    _report("Допустимые окончания пакета (hex): " + ", ".join(t.hex() for t in config.protocol.data_terminators))


class SampleSource(Protocol):
    config: Config

    def wait_next_sample(self) -> Sample: ...


def console(device: SampleSource, *, header: bool = False, input_stream=None, output_stream=None,
            report=_report) -> int:
    input_stream = sys.stdin if input_stream is None else input_stream
    output_stream = sys.stdout if output_stream is None else output_stream
    writer = csv.writer(output_stream, lineterminator="\n")
    if header:
        writer.writerow(["timestamp_utc"] + [f"ch_{c}" for c in device.config.measurement.channels])
        output_stream.flush()
    report("Приём запущен. Enter запрашивает новый полный принятый набор, не аппаратное измерение по триггеру.")
    report(HELP)
    while True:
        report("Готов. Нажмите Enter для показаний или введите q для выхода.")
        line = input_stream.readline()
        if line == "":
            report("Ввод закрыт; завершаю работу.")
            return 0
        command = line.strip().lower()
        if command in ("q", "quit", "exit"):
            return 0
        if command in ("h", "help", "?"):
            report(HELP)
            continue
        if command:
            report(f"Неизвестная команда: {command!r}. {HELP}")
            continue
        report(f"Ожидаю полный набор, не более {device.config.protocol.sample_timeout_seconds:g} с...")
        try:
            sample = device.wait_next_sample()
        except SampleTimeout as exc:
            report(str(exc))
            continue
        writer.writerow([sample.timestamp.isoformat(timespec="milliseconds")] + list(sample.values))
        output_stream.flush()
        report(f"Набор №{sample.sequence} выведен: {len(sample.values)} значений.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Минимальный приёмник EchM: одна CSV-строка по Enter.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="JSON-настройки (по умолчанию config.json рядом с проектом)")
    parser.add_argument("--port", help="Переопределить порт, например COM4")
    parser.add_argument("--trace", type=Path, help="Дописывать исходный RX/TX в локальный JSONL-файл")
    parser.add_argument("--header", action="store_true", help="Вывести CSV-заголовок перед измерениями")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--list-ports", action="store_true", help="Показать порты без подключения к прибору")
    modes.add_argument("--check-config", action="store_true", help="Проверить JSON и показать пакет настройки без открытия порта")
    modes.add_argument("--demo", action="store_true", help="Синтетические показания, без подключения к оборудованию")
    args = parser.parse_args(argv)
    if args.list_ports:
        try:
            from serial.tools import list_ports
            ports = list(list_ports.comports())
            for port in ports:
                _report(f"{port.device}: {port.description}")
            if not ports:
                _report("COM-порты не найдены.")
            return 0
        except (ImportError, OSError) as exc:
            _report(f"Не удалось получить список портов: {exc}")
            return 1
    try:
        config = Config.load(args.config)
        if args.port is not None:
            if not args.port.strip():
                raise ConfigError("--port не должен быть пустым")
            config = replace(config, serial=replace(config.serial, port=args.port.strip()))
        describe(config)
        if args.check_config:
            packet = configure_command(config.measurement)
            _report(f"JSON корректен. Команда настройки ({len(packet)} байт): {packet!r}")
            return 0
    except (ConfigError, ValueError) as exc:
        _report(f"Ошибка настроек: {exc}")
        return 2
    factory = None
    if args.demo:
        _report("ДЕМО: данные синтетические, порт не открывается; период симуляции 0.25 с.")
        factory = lambda serial_settings, read_timeout, write_timeout: SimulatedSerial(
            read_timeout, config.protocol.data_terminators[0]
        )
    device = Device(config, log=_report, trace_path=args.trace, serial_factory=factory)
    result = 0
    try:
        device.start()
        result = console(device, header=args.header)
    except KeyboardInterrupt:
        _report("\nПолучен Ctrl+C; останавливаю прибор и закрываю порт.")
        result = 130
    except (DeviceError, OSError) as exc:
        _report(f"Ошибка: {exc}")
        result = 1
    finally:
        try:
            device.close()
        except DeviceError as exc:
            _report(f"Ошибка завершения: {exc}")
            result = 1
    return result
