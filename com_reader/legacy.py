"""Isolated import of the MSVC binary layouts declared in StdAfx.h.

Only the exact StrSaveInfo (12416 bytes), StrProfileOld (8584 bytes), and
StrProfile (8620 bytes) layouts are supported. File extensions select the
container; size selects only the documented old/new .epf version. No layout
probing, serial communication, calibration, or config module is involved.
"""

from __future__ import annotations

import argparse
import codecs
import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from typing import NoReturn

__all__ = ["LegacyImportError", "import_legacy", "main"]

# MSVC: one-byte bool/char, little-endian four-byte int/float, alignment 4.
_PRM_SIZE = 12416
_PROFILE_OFFSET = 3372
_PARAMS_OFFSET = 12016
_PORT_OFFSET = 3270
_PORT_SIZE = 100
_OLD_PROFILE_SIZE = 8584
_PROFILE_SIZE = 8620
_PROFILE_NAME_OFFSET = 168
_CHANNEL_NAMES_OFFSET = 423
_REFERENCE_OFFSET = 8583  # Old structure's final padding is not a field.
_MODE_OFFSET = 8616
_FILTERS = (50, 250, 500)


class LegacyImportError(ValueError):
    """Unsupported or invalid legacy settings (messages are in Russian)."""


def _text(data: bytes, encoding: str, field: str) -> str:
    # Bytes after the first NUL and alignment padding can be uninitialized.
    try:
        return data.split(b"\0", 1)[0].decode(encoding, errors="strict")
    except (UnicodeError, LookupError) as exc:
        raise LegacyImportError(
            f"Не удалось декодировать {field} в кодировке {encoding}: {exc}"
        ) from exc


def _bools(data: bytes, field: str) -> list[int]:
    for channel, value in enumerate(data, start=1):
        if value not in (0, 1):
            raise LegacyImportError(
                f"{field}, канал {channel}: bool должен быть 0 или 1, получено {value}."
            )
    return [channel for channel, value in enumerate(data, start=1) if value == 1]


def _port(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LegacyImportError("Порт должен быть непустой строкой (например, COM4).")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise LegacyImportError("Имя порта не должно содержать управляющие символы.")
    return value


def _filter(value: int) -> int:
    if type(value) is not int or value not in _FILTERS:
        raise LegacyImportError("Частота фильтра должна быть целым числом 50, 250 или 500 Гц.")
    return value


def import_legacy(
    path: Path | str,
    *,
    port: str | None = None,
    filter_hz: int | None = None,
    encoding: str = "cp1251",
) -> dict:
    """Return schema-version-1 settings without applying legacy zero offsets.

    .epf requires explicit port and filter_hz. For .prm these arguments override
    the saved settings, but do not bypass validation of the binary fields.
    Channel delays must be 1..256: MIPSSetAdcChar emits delay - 1 as two hex
    digits. Interval and duration must be positive legacy signed int32 values.
    Reference channels are physical IDs with their reference flag enabled,
    independently of the measurement mask. All acquisition uses single mode;
    cross mode and raw offsets (whose units are unknown) remain metadata only.

    Raises LegacyImportError for unsupported/invalid data and OSError for I/O
    failures. Text is decoded strictly, using only bytes before the first NUL.
    """
    source = str(path)
    source_path = Path(path)
    suffix = source_path.suffix.lower()
    if suffix not in (".prm", ".epf"):
        raise LegacyImportError("Поддерживаются только файлы .prm и .epf.")
    if not isinstance(encoding, str):
        raise LegacyImportError("Кодировка должна быть строкой (например, cp1251).")
    try:
        codecs.lookup(encoding)
    except LookupError as exc:
        raise LegacyImportError(f"Неизвестная кодировка: {encoding}.") from exc

    data = source_path.read_bytes()
    sizes = (_PRM_SIZE,) if suffix == ".prm" else (_OLD_PROFILE_SIZE, _PROFILE_SIZE)
    if len(data) not in sizes:
        expected = " или ".join(str(size) for size in sizes)
        raise LegacyImportError(
            f"Неверный размер {suffix}: {len(data)} байт; требуется ровно {expected} байт."
        )

    zero_offsets = None
    if suffix == ".prm":
        profile = data[_PROFILE_OFFSET : _PROFILE_OFFSET + _PROFILE_SIZE]
        saved_port = _text(data[_PORT_OFFSET : _PORT_OFFSET + _PORT_SIZE], encoding, "порт")
        saved_port = _port(saved_port)
        filter_id = struct.unpack_from("<f", data, _PARAMS_OFFSET + 24 * 4)[0]
        if not math.isfinite(filter_id) or filter_id not in (0.0, 1.0, 2.0):
            raise LegacyImportError(
                f"Param[24]: ID фильтра должен быть целым 0, 1 или 2, получено {filter_id}."
            )
        saved_filter = _FILTERS[int(filter_id)]
        zero_offsets = list(struct.unpack_from("<32f", data, _PARAMS_OFFSET + 30 * 4))
        for index, value in enumerate(zero_offsets, start=30):
            if not math.isfinite(value):
                raise LegacyImportError(f"Param[{index}]: нулевое смещение должно быть конечным числом.")
        selected_port = saved_port if port is None else _port(port)
        selected_filter = saved_filter if filter_hz is None else _filter(filter_hz)
    else:
        if port is None or filter_hz is None:
            raise LegacyImportError(
                "Профиль .epf не содержит порт и частоту фильтра; укажите port и filter_hz "
                "(в CLI: --port и --filter-hz)."
            )
        profile = data
        selected_port = _port(port)
        selected_filter = _filter(filter_hz)

    channels = _bools(profile[:32], "bOnChs")
    if not channels:
        raise LegacyImportError("В профиле не включён ни один измерительный канал.")
    delays = list(struct.unpack_from("<32i", profile, 32))
    for channel, delay in enumerate(delays, start=1):
        if not 1 <= delay <= 256:
            raise LegacyImportError(
                f"ChMeasTime, канал {channel}: задержка должна быть 1..256, получено {delay}."
            )
    duration, interval = struct.unpack_from("<2i", profile, 160)
    if duration < 1:
        raise LegacyImportError(f"MeasTime должен быть положительным, получено {duration}.")
    if interval < 1:
        raise LegacyImportError(f"MeasInterval должен быть положительным, получено {interval}.")

    # StrProfile::copyProf(StrProfileOld&) initializes these fields to zero.
    references = []
    mode = 0
    if len(profile) == _PROFILE_SIZE:
        references = _bools(profile[_REFERENCE_OFFSET : _REFERENCE_OFFSET + 32], "bReferenceChs")
        mode = struct.unpack_from("<i", profile, _MODE_OFFSET)[0]
        if mode not in (0, 1):
            raise LegacyImportError(f"iAnalysisType должен быть 0 или 1, получено {mode}.")
    profile_name = _text(
        profile[_PROFILE_NAME_OFFSET : _PROFILE_NAME_OFFSET + 255], encoding, "имя профиля"
    )
    channel_names = [
        _text(
            profile[_CHANNEL_NAMES_OFFSET + index * 255 : _CHANNEL_NAMES_OFFSET + (index + 1) * 255],
            encoding,
            f"имя канала {index + 1}",
        )
        for index in range(32)
    ]
    return {
        "schema_version": 1,
        "serial": {
            "port": selected_port,
            "baudrate": 128000,
            "rtscts": False,
            "dsrdtr": False,
            "xonxoff": False,
            "rts": None,
            "dtr": None,
        },
        "measurement": {
            "mode": "single",
            "channels": channels,
            "channel_delays": delays,
            "interval_seconds": interval,
            "filter_hz": selected_filter,
        },
        "protocol": {
            "data_terminators_hex": ["3130", "0d0a", "0a"],
            "read_timeout_seconds": 0.1,
            "command_timeout_seconds": 2.0,
            "sample_timeout_seconds": max(30.0, interval * 2.0 + 5.0),
        },
        "legacy": {
            "source": source,
            "sha256": hashlib.sha256(data).hexdigest(),
            "profile_name": profile_name,
            "original_mode": "single" if mode == 0 else "cross",
            "channel_names": channel_names,
            "reference_channels": references,
            "duration_minutes": duration,
            "zero_offsets": zero_offsets,
            "zero_offsets_applied": False,
            "units": None,
        },
    }


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise LegacyImportError(f"Ошибка аргументов: {message}")


def _filter_argument(value: str) -> int:
    try:
        return _filter(int(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("укажите целую частоту 50, 250 или 500 Гц") from exc


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; no output is created until the complete import succeeds."""
    parser = _Parser(description="Перенос бинарных настроек EchM в JSON без калибровки.")
    parser.add_argument("source", metavar="SOURCE", type=Path, help="файл .prm или .epf")
    parser.add_argument("--output", metavar="FILE", required=True, type=Path, help="выходной JSON")
    parser.add_argument("--port", help="порт (обязателен для .epf)")
    parser.add_argument("--filter-hz", type=_filter_argument, help="50, 250 или 500 (обязателен для .epf)")
    parser.add_argument("--encoding", default="cp1251", help="кодировка имён, по умолчанию cp1251")
    parser.add_argument("--force", action="store_true", help="разрешить перезапись выходного файла")
    try:
        args = parser.parse_args(argv)
        if args.output.resolve() == args.source.resolve() or (
            args.output.exists() and args.source.exists() and args.output.samefile(args.source)
        ):
            raise LegacyImportError("Выходной файл не должен совпадать с исходным, даже с --force.")
        result = import_legacy(
            args.source, port=args.port, filter_hz=args.filter_hz, encoding=args.encoding
        )
        text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        # Exclusive creation prevents accidentally overwriting an existing file,
        # including one created by another process after validation.
        with args.output.open("w" if args.force else "x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        print(str(args.output))
        return 0
    except FileExistsError:
        print("Ошибка: выходной файл уже существует; для перезаписи укажите --force.", file=sys.stderr)
        return 1
    except (LegacyImportError, OSError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Импорт прерван пользователем.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
