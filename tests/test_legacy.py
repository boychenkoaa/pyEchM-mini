"""Importer tests use independently constructed MSVC layouts, never serial I/O."""

import hashlib
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

from com_reader.legacy import LegacyImportError, import_legacy, main


def _c_string(text, length=255, encoding="cp1251"):
    encoded = text.encode(encoding)
    if len(encoded) >= length:
        raise ValueError("Test string is too long")
    return encoded + bytes(length - len(encoded))


def _profile(*, old=False, mode=1, interval=20, encoding="cp1251"):
    enabled = [int(index in (0, 2, 31)) for index in range(32)]
    delays = list(range(1, 33))
    names = b"".join(_c_string(f"Электрод {index + 1}", encoding=encoding) for index in range(32))
    body = struct.pack(
        "<32B32i2i255s8160s",
        *enabled, *delays, 7, interval,
        _c_string("Профиль переноса", encoding=encoding), names,
    )
    if old:
        return body + b"\xa5"  # Nonzero old tail padding is not reference data.
    references = [int(index in (0, 15, 31)) for index in range(32)]
    return body + struct.pack("<32Bxi", *references, mode)


def _prm(*, filter_id=2.0, profile=None, port="COM17"):
    data = bytearray(12416)
    data[:255] = _c_string("Outer name, not FCurProf.NameProf")
    data[3270:3370] = _c_string(port, 100)
    data[3370:3372] = b"\xaa\xbb"  # Alignment padding is not validated as bool.
    data[3372:11992] = _profile() if profile is None else profile
    params = [0.0] * 100
    params[24] = filter_id
    params[30:62] = [index - 15.5 for index in range(32)]
    struct.pack_into("<100f", data, 12016, *params)
    return bytes(data)


class LegacyImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def write_source(self, data, name="settings.prm"):
        path = self.directory / name
        path.write_bytes(data)
        return path

    def import_epf(self, data, **kwargs):
        options = {"port": "COM8", "filter_hz": 250}
        options.update(kwargs)
        return import_legacy(self.write_source(data, "profile.epf"), **options)

    def test_prm_exact_schema_and_physical_channels(self):
        data = _prm()
        path = self.write_source(data)
        expected = {
            "schema_version": 1,
            "serial": {
                "port": "COM17", "baudrate": 128000, "rtscts": False,
                "dsrdtr": False, "xonxoff": False, "rts": None, "dtr": None,
            },
            "measurement": {
                "mode": "single", "channels": [1, 3, 32],
                "channel_delays": list(range(1, 33)), "interval_seconds": 20,
                "filter_hz": 500,
            },
            "protocol": {
                "data_terminators_hex": ["3130", "0d0a", "0a"],
                "read_timeout_seconds": 0.1, "command_timeout_seconds": 2.0,
                "sample_timeout_seconds": 45.0,
            },
            "legacy": {
                "source": str(path), "sha256": hashlib.sha256(data).hexdigest(),
                "profile_name": "Профиль переноса", "original_mode": "cross",
                "channel_names": [f"Электрод {index + 1}" for index in range(32)],
                "reference_channels": [1, 16, 32], "duration_minutes": 7,
                "zero_offsets": [index - 15.5 for index in range(32)],
                "zero_offsets_applied": False, "units": None,
            },
        }
        self.assertEqual(import_legacy(path), expected)
        self.assertEqual(import_legacy(str(path)), expected)
        self.assertEqual(json.loads(json.dumps(expected, allow_nan=False)), expected)
        self.assertEqual(path.read_bytes(), data)

    def test_old_epf_defaults_and_padding(self):
        data = _profile(old=True, interval=10)
        self.assertEqual(len(data), 8584)
        result = self.import_epf(data)
        self.assertEqual(result["legacy"]["original_mode"], "single")
        self.assertEqual(result["legacy"]["reference_channels"], [])
        self.assertIsNone(result["legacy"]["zero_offsets"])
        self.assertFalse(result["legacy"]["zero_offsets_applied"])
        self.assertIsNone(result["legacy"]["units"])
        self.assertEqual(result["serial"]["port"], "COM8")
        self.assertEqual(result["measurement"]["filter_hz"], 250)
        self.assertEqual(result["measurement"]["channels"], [1, 3, 32])
        self.assertEqual(result["protocol"]["sample_timeout_seconds"], 30.0)

    def test_new_epf_single_and_cross_metadata(self):
        for mode, name in ((0, "single"), (1, "cross")):
            with self.subTest(mode=mode):
                data = bytearray(_profile(mode=mode))
                self.assertEqual(len(data), 8620)
                data[8615] = 255  # Padding before aligned mode, not a bool.
                result = self.import_epf(data)
                self.assertEqual(result["measurement"]["mode"], "single")
                self.assertEqual(result["legacy"]["original_mode"], name)
                self.assertEqual(result["legacy"]["reference_channels"], [1, 16, 32])
                self.assertIsNone(result["legacy"]["zero_offsets"])

    def test_epf_requires_both_explicit_settings(self):
        for old in (False, True):
            path = self.write_source(_profile(old=old), "profile.epf")
            options_list: tuple[dict[str, Any], ...] = ({}, {"port": "COM1"}, {"filter_hz": 50})
            for options in options_list:
                with (
                    self.subTest(old=old, options=options),
                    self.assertRaisesRegex(LegacyImportError, "не содержит порт"),
                ):
                    import_legacy(path, **options)

    def test_prm_filter_mapping(self):
        for filter_id, hz in ((0.0, 50), (1.0, 250), (2.0, 500)):
            with self.subTest(filter_id=filter_id):
                result = import_legacy(self.write_source(_prm(filter_id=filter_id)))
                self.assertEqual(result["measurement"]["filter_hz"], hz)

    def test_prm_explicit_overrides(self):
        result = import_legacy(self.write_source(_prm()), port="COM22", filter_hz=50)
        self.assertEqual(result["serial"]["port"], "COM22")
        self.assertEqual(result["measurement"]["filter_hz"], 50)

    def test_prm_invalid_filter_ids_even_with_override(self):
        for filter_id in (-1.0, 3.0, 0.5, 1.999, float("nan"), float("inf"), -float("inf")):
            for override in (None, 500):
                with self.subTest(filter_id=filter_id, override=override):
                    path = self.write_source(_prm(filter_id=filter_id))
                    with self.assertRaisesRegex(LegacyImportError, r"Param\[24\]"):
                        import_legacy(path, filter_hz=override)

    def test_invalid_filter_arguments(self):
        path = self.write_source(_prm())
        for hz in (0, 1, 2, 51, 1000, 50.0, True, "500"):
            with self.subTest(hz=hz):
                with self.assertRaisesRegex(LegacyImportError, "Частота фильтра"):
                    import_legacy(path, filter_hz=cast(Any, hz))
                with self.assertRaisesRegex(LegacyImportError, "Частота фильтра"):
                    self.import_epf(_profile(), filter_hz=hz)

    def test_invalid_port_arguments(self):
        path = self.write_source(_prm())
        for port in ("", "   ", "COM1\0", "COM1\n", "COM1\r", 7, False):
            with self.subTest(port=port):
                with self.assertRaises(LegacyImportError):
                    import_legacy(path, port=cast(Any, port))
                with self.assertRaises(LegacyImportError):
                    self.import_epf(_profile(), port=port)

    def test_invalid_saved_port(self):
        with self.assertRaisesRegex(LegacyImportError, "Порт"):
            import_legacy(self.write_source(_prm(port="")))

    def test_prm_strict_lengths_no_profile_size_fallback(self):
        for size in (0, 8584, 8620, 12415, 12417):
            with self.subTest(size=size), self.assertRaisesRegex(LegacyImportError, "Неверный размер"):
                import_legacy(self.write_source(bytes(size)))

    def test_epf_strict_lengths(self):
        for size in (0, 168, 8583, 8585, 8619, 8621, 12416):
            with self.subTest(size=size), self.assertRaisesRegex(LegacyImportError, "Неверный размер"):
                self.import_epf(bytes(size))

    def test_extension_is_not_guessed_from_bytes(self):
        path = self.write_source(_prm(), "settings.bin")
        with self.assertRaisesRegex(LegacyImportError, "только файлы"):
            import_legacy(path)
        path = self.write_source(_prm(), "settings.PRM")
        self.assertEqual(import_legacy(path)["serial"]["port"], "COM17")
        path = self.write_source(_profile(), "profile.EPF")
        self.assertEqual(import_legacy(path, port="COM1", filter_hz=50)["measurement"]["filter_hz"], 50)

    def test_noncanonical_channel_bools_in_all_layouts(self):
        for old in (False, True):
            for channel in (0, 15, 31):
                for value in (2, 128, 255):
                    with self.subTest(old=old, channel=channel, value=value):
                        data = bytearray(_profile(old=old))
                        data[channel] = value
                        with self.assertRaisesRegex(LegacyImportError, "bOnChs"):
                            self.import_epf(data)
                        if not old:
                            with self.assertRaisesRegex(LegacyImportError, "bOnChs"):
                                import_legacy(self.write_source(_prm(profile=data)))

    def test_noncanonical_reference_bools(self):
        for channel in (0, 15, 31):
            for value in (2, 128, 255):
                with self.subTest(channel=channel, value=value):
                    data = bytearray(_profile())
                    data[8583 + channel] = value
                    with self.assertRaisesRegex(LegacyImportError, "bReferenceChs"):
                        self.import_epf(data)
                    with self.assertRaisesRegex(LegacyImportError, "bReferenceChs"):
                        import_legacy(self.write_source(_prm(profile=data)))

    def test_invalid_modes(self):
        for mode in (-1, 2, 256, 2147483647):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(LegacyImportError, "iAnalysisType"):
                    self.import_epf(_profile(mode=mode))
                with self.assertRaisesRegex(LegacyImportError, "iAnalysisType"):
                    import_legacy(self.write_source(_prm(profile=_profile(mode=mode))))

    def test_invalid_delays_including_disabled_channels(self):
        for old in (False, True):
            for channel in (0, 15, 31):
                for delay in (-1, 0, 257, 2147483647):
                    with self.subTest(old=old, channel=channel, delay=delay):
                        data = bytearray(_profile(old=old))
                        struct.pack_into("<i", data, 32 + 4 * channel, delay)
                        with self.assertRaisesRegex(LegacyImportError, "ChMeasTime"):
                            self.import_epf(data)

    def test_delay_upper_boundary(self):
        data = bytearray(_profile())
        struct.pack_into("<i", data, 32 + 31 * 4, 256)
        result = self.import_epf(data)
        self.assertEqual(result["measurement"]["channel_delays"][31], 256)

    def test_invalid_duration_and_interval(self):
        for field, offset in (("MeasTime", 160), ("MeasInterval", 164)):
            for value in (0, -1, -2147483648):
                with self.subTest(field=field, value=value):
                    data = bytearray(_profile())
                    struct.pack_into("<i", data, offset, value)
                    with self.assertRaisesRegex(LegacyImportError, field):
                        self.import_epf(data)

    def test_empty_measurement_mask_rejected(self):
        data = bytearray(_profile())
        data[:32] = bytes(32)
        with self.assertRaisesRegex(LegacyImportError, "ни один"):
            self.import_epf(data)

    def test_nonfinite_zero_offsets_rejected(self):
        for index in (30, 45, 61):
            for value in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(index=index, value=value):
                    data = bytearray(_prm())
                    struct.pack_into("<f", data, 12016 + index * 4, value)
                    with self.assertRaisesRegex(LegacyImportError, rf"Param\[{index}\]"):
                        import_legacy(self.write_source(data))

    def test_unrelated_params_are_not_interpreted(self):
        data = bytearray(_prm())
        struct.pack_into("<f", data, 12016 + 99 * 4, float("nan"))
        self.assertEqual(import_legacy(self.write_source(data))["measurement"]["filter_hz"], 500)

    def test_explicit_text_encoding(self):
        result = self.import_epf(_profile(encoding="utf-8"), encoding="utf-8")
        self.assertEqual(result["legacy"]["profile_name"], "Профиль переноса")
        self.assertEqual(result["legacy"]["channel_names"][31], "Электрод 32")

    def test_encoding_errors_are_not_silently_replaced(self):
        path = self.write_source(_prm())
        for encoding in ("no-such-codec", "utf-8", "base64_codec"):
            with self.subTest(encoding=encoding), self.assertRaises(LegacyImportError):
                import_legacy(path, encoding=encoding)
        data = bytearray(_profile())
        data[168] = 0x98  # Undefined in cp1251.
        with self.assertRaisesRegex(LegacyImportError, "имя профиля"):
            self.import_epf(data)
        data = bytearray(_profile())
        data[423 + 31 * 255] = 0x98
        with self.assertRaisesRegex(LegacyImportError, "имя канала 32"):
            self.import_epf(data)

    def test_uninitialized_bytes_after_nul_are_ignored(self):
        data = bytearray(_profile())
        data[168:423] = b"Name\0" + b"\x98" * 250
        data[423:678] = b"Ch1\0" + b"\x98" * 251
        result = self.import_epf(data)
        self.assertEqual(result["legacy"]["profile_name"], "Name")
        self.assertEqual(result["legacy"]["channel_names"][0], "Ch1")

    def test_full_width_fixed_string_is_bounded(self):
        data = bytearray(_profile())
        data[168:423] = b"P" * 255
        result = self.import_epf(data)
        self.assertEqual(result["legacy"]["profile_name"], "P" * 255)
        self.assertEqual(result["legacy"]["channel_names"][0], "Электрод 1")

    def test_missing_source_raises_io_error(self):
        with self.assertRaises(FileNotFoundError):
            import_legacy(self.directory / "missing.prm")

    def test_actual_release_prm_read_only(self):
        path = Path(__file__).resolve().parents[2] / "Release" / "EchM.prm"
        data = path.read_bytes()
        stat_before = path.stat()
        self.assertEqual(len(data), 12416)
        result = import_legacy(path)
        self.assertEqual(result["serial"]["port"], "COM4")
        self.assertEqual(result["measurement"]["mode"], "single")
        self.assertEqual(result["measurement"]["channels"], list(range(1, 11)))
        self.assertEqual(result["measurement"]["channel_delays"], [1] * 32)
        self.assertEqual(result["measurement"]["interval_seconds"], 10)
        self.assertEqual(result["measurement"]["filter_hz"], 500)
        self.assertEqual(result["legacy"]["profile_name"], "Curent")
        self.assertEqual(result["legacy"]["original_mode"], "cross")
        self.assertEqual(result["legacy"]["reference_channels"], list(range(1, 11)))
        self.assertEqual(result["legacy"]["duration_minutes"], 5)
        self.assertEqual(result["legacy"]["zero_offsets"], list(struct.unpack_from("<32f", data, 12136)))
        self.assertAlmostEqual(result["legacy"]["zero_offsets"][0], -3.60009765625)
        self.assertFalse(result["legacy"]["zero_offsets_applied"])
        self.assertIsNone(result["legacy"]["units"])
        self.assertEqual(result["legacy"]["sha256"], hashlib.sha256(data).hexdigest())
        json.dumps(result, allow_nan=False)
        self.assertEqual(path.read_bytes(), data)
        self.assertEqual(path.stat().st_mtime_ns, stat_before.st_mtime_ns)


class LegacyCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.source = self.directory / "settings.prm"
        self.source.write_bytes(_prm())
        self.output = self.directory / "settings.json"

    def run_main(self, *extra, source=None, output=None):
        stdout, stderr = io.StringIO(), io.StringIO()
        args = [str(source or self.source), "--output", str(output or self.output), *extra]
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(args)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_success_prints_output_path_and_utf8_json(self):
        code, stdout, stderr = self.run_main()
        self.assertEqual(code, 0)
        self.assertEqual(stdout, str(self.output) + "\n")
        self.assertEqual(stderr, "")
        text = self.output.read_text(encoding="utf-8")
        self.assertIn("Профиль переноса", text)
        self.assertEqual(json.loads(text), import_legacy(self.source))
        self.assertEqual(self.source.read_bytes(), _prm())

    def test_existing_output_is_preserved_without_force(self):
        self.output.write_bytes(b"existing content")
        code, stdout, stderr = self.run_main()
        self.assertNotEqual(code, 0)
        self.assertEqual(stdout, "")
        self.assertIn("--force", stderr)
        self.assertNotIn("Traceback", stderr)
        self.assertEqual(self.output.read_bytes(), b"existing content")

    def test_force_replaces_output(self):
        self.output.write_bytes(b"existing content")
        code, stdout, stderr = self.run_main("--force")
        self.assertEqual(code, 0)
        self.assertEqual(stdout, str(self.output) + "\n")
        self.assertEqual(stderr, "")
        self.assertEqual(json.loads(self.output.read_text(encoding="utf-8")), import_legacy(self.source))

    def test_force_does_not_replace_source(self):
        for extra in ((), ("--force",)):
            with self.subTest(extra=extra):
                code, stdout, stderr = self.run_main(*extra, output=self.source)
                self.assertNotEqual(code, 0)
                self.assertEqual(stdout, "")
                self.assertIn("исходным", stderr)
                self.assertEqual(self.source.read_bytes(), _prm())

    def test_hard_link_to_source_is_protected(self):
        try:
            os.link(self.source, self.output)
        except OSError as exc:
            self.skipTest(f"Hard links unavailable: {exc}")
        code, stdout, stderr = self.run_main("--force")
        self.assertNotEqual(code, 0)
        self.assertEqual(stdout, "")
        self.assertIn("исходным", stderr)
        self.assertEqual(self.source.read_bytes(), _prm())

    def test_invalid_import_creates_no_output_and_keeps_existing_output(self):
        self.source.write_bytes(b"short")
        code, stdout, stderr = self.run_main()
        self.assertNotEqual(code, 0)
        self.assertEqual(stdout, "")
        self.assertIn("Неверный размер", stderr)
        self.assertFalse(self.output.exists())
        self.output.write_bytes(b"preserved")
        code, stdout, stderr = self.run_main("--force")
        self.assertNotEqual(code, 0)
        self.assertEqual(self.output.read_bytes(), b"preserved")

    def test_missing_file_clean_error(self):
        code, stdout, stderr = self.run_main(source=self.directory / "missing.prm")
        self.assertNotEqual(code, 0)
        self.assertEqual(stdout, "")
        self.assertIn("Ошибка", stderr)
        self.assertNotIn("Traceback", stderr)
        self.assertFalse(self.output.exists())

    def test_missing_output_parent_clean_error(self):
        code, stdout, stderr = self.run_main(output=self.directory / "missing" / "config.json")
        self.assertNotEqual(code, 0)
        self.assertEqual(stdout, "")
        self.assertIn("Ошибка", stderr)
        self.assertNotIn("Traceback", stderr)

    def test_invalid_cli_filter_clean_error(self):
        for value in ("abc", "50.0", "0", "100"):
            with self.subTest(value=value):
                code, stdout, stderr = self.run_main("--filter-hz", value)
                self.assertNotEqual(code, 0)
                self.assertEqual(stdout, "")
                self.assertIn("50, 250 или 500", stderr)
                self.assertNotIn("Traceback", stderr)
                self.assertFalse(self.output.exists())

    def test_missing_required_argument_clean_error(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main([str(self.source)])
        self.assertNotEqual(code, 0)
        self.assertIn("Ошибка", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_keyboard_interrupt_returns_130(self):
        with patch("com_reader.legacy.import_legacy", side_effect=KeyboardInterrupt):
            code, stdout, stderr = self.run_main()
        self.assertEqual(code, 130)
        self.assertEqual(stdout, "")
        self.assertIn("прерван", stderr)
        self.assertFalse(self.output.exists())

    def test_epf_cli_options(self):
        source = self.directory / "profile.epf"
        source.write_bytes(_profile(encoding="utf-8"))
        code, stdout, stderr = self.run_main(source=source)
        self.assertNotEqual(code, 0)
        self.assertIn("--port", stderr)
        self.assertFalse(self.output.exists())
        code, stdout, stderr = self.run_main(
            "--port", "COM11", "--filter-hz", "50", "--encoding", "utf-8", source=source,
        )
        self.assertEqual(code, 0)
        self.assertEqual(stdout, str(self.output) + "\n")
        self.assertEqual(stderr, "")
        result = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(result["serial"]["port"], "COM11")
        self.assertEqual(result["measurement"]["filter_hz"], 50)
        self.assertEqual(result["legacy"]["profile_name"], "Профиль переноса")

    def test_module_entry_point_success_and_failure(self):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
        command = [sys.executable, "-m", "com_reader.legacy", str(self.source), "--output", str(self.output)]
        cwd = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            command, cwd=cwd, env=env, capture_output=True, encoding="utf-8", timeout=10, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(self.output))
        self.assertEqual(result.stderr, "")
        self.assertEqual(json.loads(self.output.read_text(encoding="utf-8")), import_legacy(self.source))
        result = subprocess.run(
            command, cwd=cwd, env=env, capture_output=True, encoding="utf-8", timeout=10, check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("--force", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
