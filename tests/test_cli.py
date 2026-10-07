import csv
from datetime import datetime, timezone
import io
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from com_reader.cli import console, main
from com_reader.config import Config
from com_reader.device import Sample, SampleTimeout

PROJECT = Path(__file__).resolve().parents[1]


def config():
    return Config.from_dict({
        "schema_version": 1, "serial": {"port": "TEST"},
        "measurement": {"mode": "single", "channels": [2, 5], "channel_delays": [1] * 32,
                        "interval_seconds": 1, "filter_hz": 500},
    })


class DummyDevice:
    def __init__(self, fail_once=False):
        self.config = config()
        self.calls = 0
        self.fail_once = fail_once

    def wait_next_sample(self):
        self.calls += 1
        if self.fail_once and self.calls == 1:
            raise SampleTimeout("Test sample timeout")
        return Sample(datetime(2026, 1, 1, tzinfo=timezone.utc), self.calls, (2, 5), (-1, 32767), 0)


class ConsoleTests(unittest.TestCase):
    def test_only_csv_rows_on_output_and_help_on_report(self):
        device = DummyDevice()
        output = io.StringIO()
        messages = []
        result = console(device, input_stream=io.StringIO("h\nwrong\n\n\nq\n"), output_stream=output, report=messages.append)
        self.assertEqual(result, 0)
        rows = list(csv.reader(io.StringIO(output.getvalue())))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0], ["2026-01-01T00:00:00.000+00:00", "-1", "32767"])
        self.assertEqual(device.calls, 2)
        self.assertTrue(any("Неизвестная команда" in message for message in messages))
        self.assertTrue(any("Ожидаю" in message for message in messages))

    def test_optional_header_preserves_physical_ids(self):
        output = io.StringIO()
        console(DummyDevice(), header=True, input_stream=io.StringIO("\nq\n"), output_stream=output, report=lambda m: None)
        rows = list(csv.reader(io.StringIO(output.getvalue())))
        self.assertEqual(rows[0], ["timestamp_utc", "ch_2", "ch_5"])
        self.assertEqual(len(rows), 2)

    def test_timeout_produces_no_row_and_console_remains_usable(self):
        output = io.StringIO()
        messages = []
        console(DummyDevice(fail_once=True), input_stream=io.StringIO("\n\nq\n"), output_stream=output, report=messages.append)
        self.assertEqual(len(list(csv.reader(io.StringIO(output.getvalue())))), 1)
        self.assertIn("Test sample timeout", messages)

    def test_eof_exits_without_request(self):
        device = DummyDevice()
        output = io.StringIO()
        self.assertEqual(console(device, input_stream=io.StringIO(""), output_stream=output, report=lambda m: None), 0)
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(device.calls, 0)

    def test_keyboard_interrupt_closes_device(self):
        with patch("com_reader.cli.Device") as factory, patch("com_reader.cli._report"):
            factory.return_value.start.side_effect = KeyboardInterrupt()
            self.assertEqual(main(["--demo"]), 130)
            factory.return_value.close.assert_called_once()

    def test_missing_config_is_clean_error_without_opening_device(self):
        with patch("com_reader.cli.Device") as factory, patch("com_reader.cli._report") as report:
            self.assertEqual(main(["--config", "missing-file-for-test.json"]), 2)
            factory.assert_not_called()
            self.assertTrue(report.called)


class CommandLineTests(unittest.TestCase):
    def run_cli(self, *args, input_text=""):
        return subprocess.run([sys.executable, "-m", "com_reader", *args], cwd=PROJECT,
                              input=input_text, text=True, encoding="utf-8", errors="replace",
                              capture_output=True, timeout=10, env=self.environment())

    @staticmethod
    def environment():
        import os
        return {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}

    def test_demo_two_triggers_csv_stdout_and_prompts_stderr(self):
        result = self.run_cli("--demo", input_text="\n\nq\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = list(csv.reader(io.StringIO(result.stdout)))
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(rows[0]), 11)
        counters = []
        for row in rows:
            counter = int(row[1]) - 100
            counters.append(counter)
            self.assertEqual(row[1:], [str(c * 100 + counter) for c in range(1, 11)])
        self.assertNotEqual(counters[0], counters[1])
        self.assertIn("ДЕМО", result.stderr)
        self.assertIn("Ожидаю полный набор", result.stderr)
        self.assertIn("Прибор подтвердил остановку", result.stderr)

    def test_check_config_never_connects(self):
        result = self.run_cli("--check-config")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertIn("99 байт", result.stderr)
        self.assertNotIn("открыт", result.stderr)

    def test_demo_header_and_port_override(self):
        result = self.run_cli("--demo", "--header", "--port", "TEST", input_text="\nq\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = list(csv.reader(io.StringIO(result.stdout)))
        self.assertEqual(rows[0][0], "timestamp_utc")
        self.assertEqual(rows[0][-1], "ch_10")
        self.assertEqual(len(rows), 2)
        self.assertIn("Порт: TEST", result.stderr)

    def test_demo_eof_stops_cleanly(self):
        result = self.run_cli("--demo")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertIn("Ввод закрыт", result.stderr)

    def test_invalid_cli_config_returns_two(self):
        result = self.run_cli("--config", "missing-file-for-test.json")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
