"""Config validation and loading tests; no serial hardware or DOCX dependency."""

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from com_reader.config import Config, ConfigError, MeasurementSettings
from com_reader.protocol import FrameParser, configure_command, filter_command

PROJECT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def valid_data() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "serial": {"port": "COM4"},
        "measurement": {
            "mode": "single",
            "channels": [1, 3, 32],
            "channel_delays": [1] * 32,
            "interval_seconds": 10,
            "filter_hz": 500,
        },
    }


class ConfigValidationTests(unittest.TestCase):
    def assert_invalid(self, section, key, value):
        data = valid_data()
        if section is None:
            data[key] = value
        else:
            data.setdefault(section, {})[key] = value
        with self.assertRaises(ConfigError):
            Config.from_dict(data)

    def test_minimal_config_defaults_and_physical_channel_tuples(self):
        config = Config.from_dict(valid_data())
        self.assertEqual(config.serial.port, "COM4")
        self.assertEqual(config.serial.baudrate, 128000)
        self.assertFalse(config.serial.rtscts)
        self.assertFalse(config.serial.dsrdtr)
        self.assertFalse(config.serial.xonxoff)
        self.assertIsNone(config.serial.rts)
        self.assertIsNone(config.serial.dtr)
        measurement = config.measurement
        self.assertIsInstance(measurement, MeasurementSettings)
        self.assertEqual(measurement.channels, (1, 3, 32))
        self.assertIsInstance(measurement.channels, tuple)
        self.assertEqual(measurement.channel_delays, (1,) * 32)
        self.assertIsInstance(measurement.channel_delays, tuple)
        self.assertIs(type(measurement.interval_seconds), int)
        self.assertIs(type(measurement.filter_hz), int)
        self.assertEqual(measurement.interval_seconds, 10)
        self.assertEqual(measurement.filter_hz, 500)
        self.assertEqual(measurement.mode, "single")
        self.assertEqual(config.protocol.data_terminators, (b"10", b"\r\n", b"\n"))
        self.assertEqual(config.protocol.read_timeout_seconds, 0.1)
        self.assertEqual(config.protocol.command_timeout_seconds, 2.0)
        self.assertEqual(config.protocol.sample_timeout_seconds, 30.0)
        self.assertEqual(config.legacy, {})

    def test_serial_explicit_settings_and_port_trimming(self):
        data = valid_data()
        data["serial"] = {
            "port": "  COM12 \t", "baudrate": 9600,
            "rtscts": True, "dsrdtr": True, "xonxoff": True,
            "rts": False, "dtr": True,
        }
        serial = Config.from_dict(data).serial
        self.assertEqual(serial.port, "COM12")
        self.assertEqual(serial.baudrate, 9600)
        self.assertTrue(serial.rtscts)
        self.assertTrue(serial.dsrdtr)
        self.assertTrue(serial.xonxoff)
        self.assertIs(serial.rts, False)
        self.assertIs(serial.dtr, True)

    def test_all_channel_and_frequency_boundaries(self):
        for channels in ([1], [32], list(range(1, 33))):
            for frequency in (50, 250, 500):
                with self.subTest(channels=channels, frequency=frequency):
                    data = valid_data()
                    data["measurement"]["channels"] = channels
                    data["measurement"]["filter_hz"] = frequency
                    measurement = Config.from_dict(data).measurement
                    self.assertEqual(measurement.channels, tuple(channels))
                    self.assertEqual(measurement.filter_hz, frequency)
                    self.assertEqual(len(configure_command(measurement)), 99)

    def test_channels_do_not_alias_the_input_list(self):
        data = valid_data()
        config = Config.from_dict(data)
        data["measurement"]["channels"][0] = 2
        data["measurement"]["channel_delays"][0] = 256
        self.assertEqual(config.measurement.channels, (1, 3, 32))
        self.assertEqual(config.measurement.channel_delays, (1,) * 32)

    def test_sparse_delays_include_all_32_physical_positions(self):
        data = valid_data()
        data["measurement"]["channel_delays"] = [1, 0, 2] + [0] * 28 + [256]
        data["measurement"]["interval_seconds"] = 13
        settings = Config.from_dict(data).measurement
        self.assertEqual(settings.channels, (1, 3, 32))
        self.assertEqual(settings.channel_delays[0], 1)
        self.assertEqual(settings.channel_delays[1], 0)
        self.assertEqual(settings.channel_delays[2], 2)
        self.assertEqual(settings.channel_delays[31], 256)
        self.assertEqual(settings.minimum_cycle_seconds, 10)
        expected = (b":2" + b"000001" + b"00" * 28 + b"FF"
                    + b"80000005" + b"00000000" + b"001E"
                    + b"000000000000\n")
        self.assertEqual(configure_command(settings), expected)
        self.assertEqual(len(expected), 99)

    def test_legacy_timing_truncation_and_disabled_channel_exclusion(self):
        for delay, minimum in ((24, 0), (25, 1), (26, 1), (49, 1)):
            with self.subTest(delay=delay):
                data = valid_data()
                data["measurement"]["channels"] = [1]
                data["measurement"]["channel_delays"] = [delay] + [256] * 31
                data["measurement"]["interval_seconds"] = 1
                settings = Config.from_dict(data).measurement
                self.assertEqual(settings.minimum_cycle_seconds, minimum)
                self.assertEqual(configure_command(settings)[82:86],
                                 b"000A" if minimum == 0 else b"0001")
        data = valid_data()
        data["measurement"]["channels"] = [1]
        data["measurement"]["channel_delays"] = [50] + [0] * 31
        data["measurement"]["interval_seconds"] = 1
        with self.assertRaises(ConfigError):
            Config.from_dict(data)
        data["measurement"]["interval_seconds"] = 2
        self.assertEqual(Config.from_dict(data).measurement.minimum_cycle_seconds, 2)

    def test_maximum_cycle_and_interval(self):
        data = valid_data()
        data["measurement"]["channels"] = list(range(1, 33))
        data["measurement"]["channel_delays"] = [256] * 32
        data["measurement"]["interval_seconds"] = 327
        settings = Config.from_dict(data).measurement
        self.assertEqual(settings.minimum_cycle_seconds, 327)
        self.assertEqual(configure_command(settings)[82:86], b"0001")
        data["measurement"]["interval_seconds"] = 326
        with self.assertRaises(ConfigError):
            Config.from_dict(data)
        data = valid_data()
        data["measurement"]["interval_seconds"] = 6553
        self.assertEqual(configure_command(Config.from_dict(data).measurement)[82:86],
                         b"FFFA")

    def test_unknown_keys_in_every_strict_section(self):
        for section in (None, "serial", "measurement", "protocol"):
            with self.subTest(section=section):
                self.assert_invalid(section, "typo", 1)

    def test_root_sections_must_be_objects(self):
        for root in (None, [], "config", 1, True):
            with self.subTest(root=root), self.assertRaises(ConfigError):
                Config.from_dict(root)
        for section in ("serial", "measurement", "protocol", "legacy"):
            for value in (None, [], "section", 1, False):
                with self.subTest(section=section, value=value):
                    self.assert_invalid(None, section, value)

    def test_required_fields_cannot_be_omitted(self):
        fields = ((None, "schema_version"), (None, "serial"),
                  (None, "measurement"), ("serial", "port"),
                  ("measurement", "mode"), ("measurement", "channels"),
                  ("measurement", "channel_delays"),
                  ("measurement", "interval_seconds"),
                  ("measurement", "filter_hz"))
        for section, key in fields:
            with self.subTest(section=section, key=key):
                data = valid_data()
                del (data if section is None else data[section])[key]
                with self.assertRaises(ConfigError):
                    Config.from_dict(data)

    def test_schema_version_rejects_bool_float_and_unsupported_versions(self):
        for value in (True, False, 1.0, "1", None, 0, 2, -1):
            with self.subTest(value=value):
                self.assert_invalid(None, "schema_version", value)

    def test_serial_port_validation(self):
        for value in (None, "", " \t\r\n", 4, True, [], {}):
            with self.subTest(value=value):
                self.assert_invalid("serial", "port", value)

    def test_baudrate_range_and_bool_as_integer_rejection(self):
        for value in (0, -1, 4_000_001, True, False, 128000.0, "128000", None):
            with self.subTest(value=value):
                self.assert_invalid("serial", "baudrate", value)
        for value in (1, 4_000_000):
            data = valid_data()
            data["serial"]["baudrate"] = value
            self.assertEqual(Config.from_dict(data).serial.baudrate, value)

    def test_serial_flags_require_actual_booleans(self):
        for key in ("rtscts", "dsrdtr", "xonxoff", "rts", "dtr"):
            for value in (0, 1, "false", "true", [], {}):
                with self.subTest(key=key, value=value):
                    self.assert_invalid("serial", key, value)
        for key in ("rtscts", "dsrdtr", "xonxoff"):
            with self.subTest(key=key, value=None):
                self.assert_invalid("serial", key, None)
        for key in ("rts", "dtr"):
            for value in (None, True, False):
                data = valid_data()
                data["serial"][key] = value
                self.assertIs(getattr(Config.from_dict(data).serial, key), value)

    def test_cross_and_other_modes_rejected(self):
        for value in ("cross", "Cross", "single ", "Single", "", None, 0, True):
            with self.subTest(value=value):
                self.assert_invalid("measurement", "mode", value)

    def test_channels_require_sorted_unique_physical_integer_ids(self):
        invalid = ([], [1, 1], [1, 3, 3], [3, 1], [1, 32, 3], [0], [33],
                   [-1], [True], [False], [1.0], ["1"], [None], None, "1", (1,))
        for value in invalid:
            with self.subTest(value=value):
                self.assert_invalid("measurement", "channels", value)

    def test_delay_shape_and_integer_range_on_enabled_and_disabled_channels(self):
        for value in (None, [], [1] * 31, [1] * 33, "1", (1,) * 32):
            with self.subTest(value=value):
                self.assert_invalid("measurement", "channel_delays", value)
        for index in (0, 1, 31):
            for value in (-1, 257, True, False, 1.0, "1", None):
                with self.subTest(index=index, value=value):
                    delays: list[Any] = [1] * 32
                    delays[index] = value
                    self.assert_invalid("measurement", "channel_delays", delays)
        for index in (0, 2, 31):
            with self.subTest(enabled_zero=index):
                delays = [1] * 32
                delays[index] = 0
                self.assert_invalid("measurement", "channel_delays", delays)

    def test_interval_range_and_bool_as_integer_rejection(self):
        for value in (0, -1, 6554, True, False, 10.0, "10", None):
            with self.subTest(value=value):
                self.assert_invalid("measurement", "interval_seconds", value)

    def test_frequency_is_one_of_three_integer_values(self):
        for value in (0, 49, 51, 100, 249, 251, 499, 501,
                      True, False, 500.0, "500", None):
            with self.subTest(value=value):
                self.assert_invalid("measurement", "filter_hz", value)

    def test_hex_terminators_decode_in_configured_order(self):
        data = valid_data()
        data["protocol"] = {"data_terminators_hex": ["0A", "31 30", "0D0a"]}
        self.assertEqual(Config.from_dict(data).protocol.data_terminators,
                         (b"\n", b"10", b"\r\n"))
        data["protocol"]["data_terminators_hex"] = ["0123456789abcdef"]
        self.assertEqual(Config.from_dict(data).protocol.data_terminators,
                         (b"\x01\x23\x45\x67\x89\xab\xcd\xef",))

    def test_hex_terminator_container_and_contents_validation(self):
        invalid = (None, [], "3130", ("3130",), [None], [True], [10],
                   [b"10"], [""], [" "], ["1"], ["0g"], ["0x10"],
                   ["000102030405060708"], ["0a", "0A"], ["31 30", "3130"])
        for value in invalid:
            with self.subTest(value=value):
                self.assert_invalid("protocol", "data_terminators_hex", value)

    def test_ambiguous_prefix_endings_rejected_in_either_order(self):
        for terms in (["31", "3130"], ["3130", "31"],
                      ["0d", "0d0a"], ["0d0a", "0d"],
                      ["0a", "0a00"], ["00", "0000", "000000"]):
            with self.subTest(terms=terms):
                self.assert_invalid("protocol", "data_terminators_hex", terms)
        # LF is a suffix of CRLF, not an ambiguous prefix.
        data = valid_data()
        data["protocol"] = {"data_terminators_hex": ["0d0a", "0a"]}
        self.assertEqual(Config.from_dict(data).protocol.data_terminators,
                         (b"\r\n", b"\n"))

    def test_timeout_values_must_be_positive_finite_numbers_not_booleans(self):
        for key in ("read_timeout_seconds", "command_timeout_seconds",
                    "sample_timeout_seconds"):
            for value in (0, -1, -0.1, float("nan"), float("inf"),
                          float("-inf"), True, False, "1", None):
                with self.subTest(key=key, value=value):
                    self.assert_invalid("protocol", key, value)
        data = valid_data()
        data["protocol"] = {"read_timeout_seconds": 1,
                            "command_timeout_seconds": 1,
                            "sample_timeout_seconds": 2}
        protocol = Config.from_dict(data).protocol
        self.assertEqual(protocol.read_timeout_seconds, 1.0)
        self.assertEqual(protocol.command_timeout_seconds, 1.0)
        self.assertEqual(protocol.sample_timeout_seconds, 2.0)
        self.assertIs(type(protocol.read_timeout_seconds), float)

    def test_read_timeout_cannot_exceed_command_or_sample_timeout(self):
        for values in ({"read_timeout_seconds": 2.1},
                       {"read_timeout_seconds": 0.2, "command_timeout_seconds": 0.1},
                       {"read_timeout_seconds": 0.2, "sample_timeout_seconds": 0.1}):
            with self.subTest(values=values):
                data = valid_data()
                data["protocol"] = values
                with self.assertRaises(ConfigError):
                    Config.from_dict(data)
        data = valid_data()
        data["protocol"] = {"read_timeout_seconds": 2,
                            "command_timeout_seconds": 2,
                            "sample_timeout_seconds": 2}
        self.assertEqual(Config.from_dict(data).protocol.read_timeout_seconds, 2.0)

    def test_legacy_zero_offsets_are_metadata_only_and_true_rejected(self):
        metadata = {"source": "old.prm", "original_mode": "cross",
                    "zero_offsets": [123.5] * 32, "zero_offsets_applied": False,
                    "profile_name": "Curent"}
        data = valid_data()
        data["legacy"] = metadata
        config = Config.from_dict(data)
        self.assertEqual(config.legacy, metadata)
        self.assertEqual(config.measurement.mode, "single")
        self.assertEqual(configure_command(config.measurement),
                         configure_command(Config.from_dict(valid_data()).measurement))
        for value in (True, 0, 1, None, "false", [], {}):
            with self.subTest(applied=value):
                self.assert_invalid("legacy", "zero_offsets_applied", value)
        data["legacy"] = {"zero_offsets": [1.0] * 32}
        self.assertEqual(Config.from_dict(data).legacy, data["legacy"])


class ConfigLoadTests(unittest.TestCase):
    def test_load_utf8_json_path_and_string_with_optional_bom(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            data = valid_data()
            data["legacy"] = {"profile_name": "Профиль"}
            for encoding in ("utf-8", "utf-8-sig"):
                for argument in (path, str(path)):
                    with self.subTest(encoding=encoding, argument_type=type(argument)):
                        path.write_text(json.dumps(data, ensure_ascii=False), encoding=encoding)
                        before = path.read_bytes()
                        self.assertEqual(Config.load(argument), Config.from_dict(data))
                        self.assertEqual(path.read_bytes(), before)

    def test_load_io_encoding_and_json_errors_are_config_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            with self.assertRaises(ConfigError):
                Config.load(path)
            with self.assertRaises(ConfigError):
                Config.load(directory)
            for content in (b"", b"{", b'{"serial":}', b"\xff\xfe\xff"):
                with self.subTest(content=content):
                    path.write_bytes(content)
                    with self.assertRaises(ConfigError):
                        Config.load(path)
                    self.assertEqual(path.read_bytes(), content)

    def test_load_valid_json_still_requires_valid_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for data in (None, [], {}, {"schema_version": True}):
                with self.subTest(data=data):
                    path.write_text(json.dumps(data), encoding="utf-8")
                    with self.assertRaises(ConfigError):
                        Config.load(path)

    def test_actual_migrated_config_integration_if_present(self):
        path = PROJECT / "config.json"
        if not path.is_file():
            self.skipTest("Actual migrated config.json is not available")
        before = path.read_bytes()
        config = Config.load(path)
        self.assertEqual(config.measurement.mode, "single")
        self.assertEqual(config.measurement.channels, tuple(range(1, 11)))
        self.assertEqual(config.measurement.channel_delays, (1,) * 32)
        self.assertEqual(config.measurement.interval_seconds, 10)
        self.assertEqual(config.measurement.filter_hz, 500)
        self.assertEqual(config.legacy["original_mode"], "cross")
        self.assertIs(config.legacy["zero_offsets_applied"], False)
        self.assertEqual(len(config.legacy["zero_offsets"]), 32)
        self.assertEqual(len(config.legacy["sha256"]), 64)
        expected = (b":2" + b"00" * 32 + b"000003FF" + b"00000000"
                    + b"0064" + b"000000000000\n")
        self.assertEqual(configure_command(config.measurement), expected)
        self.assertEqual(len(expected), 99)
        self.assertEqual(filter_command(config.measurement.filter_hz), b":5\x03")
        sample = (FIXTURES / "cross10.txt").read_bytes().splitlines()
        parser = FrameParser(len(config.measurement.channels),
                             config.protocol.data_terminators)
        frames = parser.feed(b"".join(sample), received_ns=100)
        self.assertEqual(len(frames), 70)
        self.assertEqual([f.reference_id for f in frames], list(range(10)) * 7)
        self.assertEqual(frames[0].values,
                         (4990, -1311, -1255, -1675, 84, -1775, -1577, -1769,
                          -1237, -2363))
        self.assertEqual([f.raw for f in frames], sample)
        self.assertEqual(parser.invalid_frames, 0)
        self.assertEqual(parser.discarded_bytes, 0)
        self.assertEqual(parser.buffered_bytes, 0)
        self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
