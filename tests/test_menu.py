"""Menu regressions: scripted UI, synthetic transports, and bounded library prompts."""

import asyncio
import csv
import importlib.util
import io
import json
import tempfile
import threading
import unittest
from collections import deque
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from com_reader.config import Config, ConfigError
from com_reader.device import Device, DeviceError, Sample, SampleTimeout
from com_reader.menu import MenuApplication, QuestionaryUI, available_ports, main
from com_reader.protocol import STOP_COMMAND
from com_reader.simulation import SimulatedSerial


HAS_QUESTIONARY = importlib.util.find_spec("questionary") is not None
WAIT_SECONDS = 2
UTC = timezone.utc


class MenuTestCase(unittest.TestCase):
    def enterContext(self, context):
        # TestCase.enterContext is available only from Python 3.11.
        result = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        return result


def config_data(port="TEST-NO-HARDWARE"):
    return {
        "schema_version": 1,
        "serial": {"port": port},
        "measurement": {
            "mode": "single", "channels": [2, 5], "channel_delays": [1] * 32,
            "interval_seconds": 1, "filter_hz": 500,
        },
        "protocol": {
            "read_timeout_seconds": 0.01, "command_timeout_seconds": 0.1,
            "sample_timeout_seconds": 0.5,
        },
    }


def sample(sequence=7):
    return Sample(datetime(2026, 1, 2, 3, 4, 5, 123000, tzinfo=UTC),
                  sequence, (2, 5), (-32768, 32767), 0)


class FakeUI:
    """Reject invented selection values and fail rather than wait for missing input."""

    def __init__(self, selections=(), texts=(), confirmations=()):
        self.selections = deque(selections)
        self.texts = deque(texts)
        self.confirmations = deque(confirmations)
        self.select_calls = []
        self.text_calls = []
        self.confirm_calls = []
        self.messages = []
        self.closed = False

    @staticmethod
    def answer(queue, prompt):
        if not queue:
            raise AssertionError(f"Unscripted prompt: {prompt}")
        answer = queue.popleft()
        if callable(answer):
            answer = answer()
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def select(self, title: str, choices: list[tuple[str, str | None]]) -> str | None:
        if not isinstance(choices, list) or not choices:
            raise AssertionError("select requires a nonempty list of choices")
        for label, value in choices:
            if not isinstance(label, str) or not (value is None or isinstance(value, str)):
                raise AssertionError(f"Invalid choice: {(label, value)!r}")
        self.select_calls.append((title, list(choices)))
        answer = self.answer(self.selections, title)
        if not (answer is None or isinstance(answer, str)) or answer not in [value for label, value in choices]:
            raise AssertionError(f"{answer!r} is not offered by {title}: {choices!r}")
        return answer

    def text(self, title: str, default: str = "") -> str | None:
        self.text_calls.append((title, default))
        answer = self.answer(self.texts, title)
        if not (answer is None or isinstance(answer, str)):
            raise AssertionError("text requires str or None")
        return answer

    def confirm(self, title: str) -> bool:
        self.confirm_calls.append(title)
        answer = self.answer(self.confirmations, title)
        if type(answer) is not bool:
            raise AssertionError("confirm requires bool")
        return answer

    def report(self, message: str) -> None:
        self.messages.append(message)

    def close(self):
        self.closed = True

    def assert_consumed(self, test):
        test.assertEqual(list(self.selections), [])
        test.assertEqual(list(self.texts), [])
        test.assertEqual(list(self.confirmations), [])


class DeviceMock:
    """Controllable lifecycle; every factory kwarg and resource action is recorded."""

    def __init__(self, *, start_error=None, close_error=None, running_after_close=False,
                 samples=()):
        self.start_error = start_error
        self.close_error = close_error
        self.running_after_close = running_after_close
        self.samples = deque(samples)
        self.calls = []
        self.is_running = False
        self.resources_closed = False
        self.config: Config | None = None
        self.kwargs = {}

    def start(self):
        self.calls.append("start")
        self.is_running = True
        self.resources_closed = False
        if self.start_error is not None:
            raise self.start_error

    def close(self):
        self.calls.append("close")
        self.is_running = self.running_after_close
        self.resources_closed = not self.is_running
        if self.close_error is not None:
            raise self.close_error

    def wait_next_sample(self) -> Sample:
        self.calls.append("sample")
        answer = FakeUI.answer(self.samples, "sample")
        if not isinstance(answer, Sample):
            raise AssertionError("wait_next_sample requires Sample")
        return answer


class DeviceFactory:
    def __init__(self, *devices: DeviceMock):
        self.devices = deque(devices)
        self.calls = []

    def __call__(self, config: Config, **kwargs) -> DeviceMock:
        self.calls.append((config, kwargs))
        if not self.devices:
            raise AssertionError("Unexpected device construction")
        device = self.devices.popleft()
        device.config, device.kwargs = config, kwargs
        return device


class ScriptedMenuApplication(MenuApplication):
    ui: FakeUI
    device_factory: DeviceFactory


class MenuTests(MenuTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.config_path = self.directory / "settings.json"
        self.config_path.write_text(json.dumps(config_data()), encoding="utf-8")
        self.original_json = self.config_path.read_bytes()
        self.config = Config.load(self.config_path)
        self.output = io.StringIO()
        # Any accidental fallback to hardware fails before importing/opening serial.
        self.hardware = self.enterContext(patch(
            "com_reader.device.open_serial", side_effect=AssertionError("Real COM forbidden")))

    def app(self, ui=None, factory=None, **kwargs):
        ui = FakeUI() if ui is None else ui
        factory = DeviceFactory() if factory is None else factory
        return ScriptedMenuApplication(self.config, ui, config_path=self.config_path,
                                       device_factory=factory, ports_provider=list,
                                       output=self.output, **kwargs)

    def assert_message(self, ui, text):
        self.assertTrue(any(text in message for message in ui.messages), ui.messages)

    def test_launch_never_constructs_device_or_enumerates_ports(self):
        ui = FakeUI([None])
        app = self.app(ui)
        app.ports_provider = Mock(side_effect=AssertionError("Unexpected enumeration"))
        self.assertEqual(app.run(), 0)
        self.assertEqual(app.device_factory.calls, [])
        app.ports_provider.assert_not_called()
        self.hardware.assert_not_called()
        self.assertEqual(self.output.getvalue(), "")
        self.assert_message(ui, "не открывает порт автоматически")
        ui.assert_consumed(self)

    def test_root_and_all_submenus_back_return_to_root(self):
        keys = ["connection", "settings", "measurements", "diagnostics"]
        ui = FakeUI([item for key in keys for item in (key, None)] + [None])
        app = self.app(ui)
        self.assertEqual(app.run(), 0)
        self.assertEqual(len(ui.select_calls), 9)
        for index in (0, 2, 4, 6, 8):
            title, choices = ui.select_calls[index]
            self.assertIn("не подключено", title)
            self.assertEqual([value for label, value in choices], keys + [None])
        for index in (1, 3, 5, 7):
            self.assertEqual(ui.select_calls[index][1][-1], ("Назад", None))
        self.assertEqual(app.device_factory.calls, [])
        ui.assert_consumed(self)

    def test_submenu_action_stays_in_submenu_and_packet_is_ui_only(self):
        ui = FakeUI(["settings", "show", None, "diagnostics", "packet", None, None])
        app = self.app(ui)
        self.assertEqual(app.run(), 0)
        self.assertEqual(ui.select_calls[1][0], ui.select_calls[2][0])
        self.assertEqual(ui.select_calls[4][0], ui.select_calls[5][0])
        self.assert_message(ui, "99 байт")
        self.assert_message(ui, "2, 5")
        self.assertEqual(self.output.getvalue(), "")
        ui.assert_consumed(self)

    def test_fake_ui_rejects_labels_and_unknown_values(self):
        with self.assertRaisesRegex(AssertionError, "not offered"):
            FakeUI(["Назад"]).select("test", [("Назад", None)])

    def test_usb_com_listing_reports_description_and_hwid(self):
        ports = [Mock(device="COM9", description="USB bridge", hwid="USB VID:PID=1234:5678"),
                 Mock(device="COM2", description="UART", hwid="PNP0501")]
        app = self.app()
        app.ports_provider = Mock(return_value=ports)
        self.assertIs(app.list_ports(), ports)
        for port in ports:
            self.assert_message(app.ui, port.device)
            self.assert_message(app.ui, port.description)
            self.assert_message(app.ui, "HWID: " + port.hwid)
        self.assertEqual(self.output.getvalue(), "")

    def test_no_ports_reports_usb_driver_hint(self):
        app = self.app()
        self.assertEqual(app.list_ports(), [])
        self.assert_message(app.ui, "USB")
        self.assert_message(app.ui, "драйвер")

    def test_available_ports_sorts_without_opening_transport(self):
        # Keep this test stdlib-only even when pyserial is absent.
        ports = [Mock(device="COM9"), Mock(device="COM2")]
        list_ports = Mock()
        list_ports.comports.return_value = ports
        serial = Mock()
        serial.tools.list_ports = list_ports
        with patch.dict("sys.modules", {"serial": serial, "serial.tools": serial.tools,
                                        "serial.tools.list_ports": list_ports}):
            self.assertEqual(available_ports(), ports[::-1])
        list_ports.comports.assert_called_once_with()
        serial.Serial.assert_not_called()

    def test_choose_detected_port_changes_only_current_session(self):
        ui = FakeUI(["COM9"])
        app = self.app(ui)
        app.ports_provider = lambda: [Mock(device="COM9", description="USB", hwid="test-hwid")]
        app.choose_port()
        self.assertEqual(app.config.serial.port, "COM9")
        self.assertEqual(self.config.serial.port, "TEST-NO-HARDWARE")
        self.assertEqual(app.config_path, self.config_path)
        self.assertEqual(self.config_path.read_bytes(), self.original_json)
        self.assertIn(("COM9 — USB", "COM9"), ui.select_calls[0][1])
        self.assertEqual(app.device_factory.calls, [])
        self.assert_message(ui, "JSON не перезаписан")

    def test_manual_port_trims_whitespace_without_json_write(self):
        ui = FakeUI(["manual"], ["  COM42  "])
        app = self.app(ui)
        app.choose_port()
        self.assertEqual(app.config.serial.port, "COM42")
        self.assertEqual(ui.text_calls[0][1], self.config.serial.port)
        self.assertEqual(self.config_path.read_bytes(), self.original_json)

    def test_manual_port_rejects_blank_and_control_characters(self):
        for name in ("", "   ", "\t\r\n", "COM\x00X", "COM\n4"):
            with self.subTest(name=name):
                ui = FakeUI(["manual"], [name])
                app = self.app(ui)
                app.choose_port()
                self.assertIs(app.config, self.config)
                self.assert_message(ui, "непустым и без управляющих символов")
        self.assertEqual(self.config_path.read_bytes(), self.original_json)

    def test_port_back_or_manual_cancel_keeps_settings(self):
        for ui in (FakeUI([None]), FakeUI(["manual"], [None])):
            with self.subTest(ui=ui):
                app = self.app(ui)
                app.choose_port()
                self.assertIs(app.config, self.config)
                ui.assert_consumed(self)

    def test_choose_port_blocked_during_session_before_prompts(self):
        app = self.app()
        app.device = DeviceMock()
        app.ports_provider = Mock()
        app.choose_port()
        app.ports_provider.assert_not_called()
        self.assertEqual(app.ui.select_calls, [])
        self.assertEqual(app.ui.text_calls, [])
        self.assertIs(app.config, self.config)
        self.assert_message(app.ui, "Сначала отключите")

    def test_real_connect_declined_does_not_call_factory(self):
        ui = FakeUI(confirmations=[False])
        factory = DeviceFactory()
        app = self.app(ui, factory)
        app.connect()
        self.assertEqual(len(ui.confirm_calls), 1)
        self.assertIn("STOP", ui.confirm_calls[0])
        self.assertIn("START", ui.confirm_calls[0])
        self.assertEqual(factory.calls, [])
        self.assertIsNone(app.device)
        self.assert_message(ui, "порт не открывался")

    def test_real_connect_confirmed_passes_config_log_trace_and_no_simulator(self):
        ui = FakeUI(confirmations=[True])
        device = DeviceMock()
        factory = DeviceFactory(device)
        trace = self.directory / "wire.jsonl"
        app = self.app(ui, factory, trace_path=trace)
        app.connect()
        self.assertEqual(len(factory.calls), 1)
        self.assertIs(device.config, self.config)
        self.assertEqual(device.kwargs, {"log": ui.report, "trace_path": trace,
                                         "serial_factory": None})
        self.assertEqual(device.calls, ["start"])
        self.assertIs(app.device, device)
        self.assertFalse(app.demo)
        device.kwargs["log"]("worker diagnostic")
        self.assertIn("worker diagnostic", ui.messages)
        app.disconnect()
        self.hardware.assert_not_called()

    def test_demo_passes_simulated_serial_factory_and_never_confirms(self):
        device = DeviceMock()
        app = self.app(factory=DeviceFactory(device))
        app.connect(demo=True)
        self.assertEqual(app.ui.confirm_calls, [])
        self.assertTrue(app.demo)
        self.assertEqual(set(device.kwargs), {"log", "trace_path", "serial_factory"})
        factory = device.kwargs["serial_factory"]
        self.assertTrue(callable(factory))
        transport = factory(self.config.serial, 0.037, 0.123)
        self.addCleanup(transport.close)
        self.assertIs(type(transport), SimulatedSerial)
        self.assertEqual(transport.timeout, 0.037)
        self.assertEqual(transport.terminator, self.config.protocol.data_terminators[0])
        self.assert_message(app.ui, "только синтетические")
        self.assertEqual(self.output.getvalue(), "")
        app.disconnect()
        self.hardware.assert_not_called()

    def test_sample_is_one_csv_row_stdout_and_status_only_on_ui(self):
        device = DeviceMock(samples=[sample()])
        ui = FakeUI(["connection", "demo", None, "measurements", "sample", None, None])
        app = self.app(ui, DeviceFactory(device))
        self.assertEqual(app.run(), 0)
        self.assertEqual(list(csv.reader(io.StringIO(self.output.getvalue()))),
                         [["2026-01-02T03:04:05.123+00:00", "-32768", "32767"]])
        self.assertEqual(device.calls, ["start", "sample", "close"])
        self.assert_message(ui, "Набор №7: 2 значений")
        self.assert_message(ui, "программный запрос")
        self.assertTrue(device.resources_closed)
        ui.assert_consumed(self)

    def test_acquire_without_session_has_no_csv(self):
        app = self.app()
        app.acquire()
        self.assertEqual(self.output.getvalue(), "")
        self.assert_message(app.ui, "Сначала выберите")

    def test_sample_timeout_has_no_csv_keeps_session_and_can_retry(self):
        device = DeviceMock(samples=[SampleTimeout("sample timed out"), sample()])
        app = self.app(factory=DeviceFactory(device))
        app.connect(demo=True)
        app.acquire()
        self.assertEqual(self.output.getvalue(), "")
        self.assertIs(app.device, device)
        self.assertTrue(device.is_running)
        self.assertEqual(app.exit_code, 0)
        self.assertEqual(device.calls, ["start", "sample"])
        self.assert_message(app.ui, "sample timed out")
        app.acquire()
        self.assertEqual(len(list(csv.reader(io.StringIO(self.output.getvalue())))), 1)
        app.disconnect()

    def test_load_json_validates_and_preserves_original_files(self):
        path = self.directory / "other.json"
        path.write_text(json.dumps(config_data("COM77")), encoding="utf-8")
        before = path.read_bytes()
        ui = FakeUI(texts=["  " + str(path) + "  "])
        app = self.app(ui)
        app.load_config()
        self.assertEqual(app.config, Config.load(path))
        self.assertEqual(app.config_path, path)
        self.assertEqual(ui.text_calls[0][1], str(self.config_path))
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(self.config_path.read_bytes(), self.original_json)
        self.assert_message(ui, "JSON загружен")
        self.assertEqual(app.device_factory.calls, [])

    def test_invalid_json_keeps_old_config_and_path(self):
        invalid = config_data()
        invalid["measurement"]["channels"] = [5, 2]
        path = self.directory / "invalid.json"
        for text in ("not-json", json.dumps(invalid), json.dumps({"schema_version": 1})):
            with self.subTest(text=text):
                path.write_text(text, encoding="utf-8")
                app = self.app(FakeUI(texts=[str(path)]))
                with self.assertRaises(ConfigError):
                    app.load_config()
                self.assertIs(app.config, self.config)
                self.assertEqual(app.config_path, self.config_path)
                self.assertEqual(path.read_text(encoding="utf-8"), text)

    def test_load_invalid_json_submenu_reports_error_and_allows_back(self):
        ui = FakeUI(["settings", "load", None, None], [str(self.directory / "missing.json")])
        app = self.app(ui)
        self.assertEqual(app.run(), 0)
        self.assertIs(app.config, self.config)
        self.assert_message(ui, "Ошибка:")
        self.assertEqual(self.output.getvalue(), "")
        ui.assert_consumed(self)

    def test_load_json_blocked_while_connected_without_reading_or_prompt(self):
        app = self.app()
        app.device = DeviceMock()
        with patch("com_reader.menu.Config.load") as load:
            app.load_config()
        load.assert_not_called()
        self.assertEqual(app.ui.text_calls, [])
        self.assertIs(app.config, self.config)
        self.assertEqual(app.config_path, self.config_path)

    def test_load_cancel_or_blank_keeps_config(self):
        for name in (None, "", "  "):
            with self.subTest(name=name):
                app = self.app(FakeUI(texts=[name]))
                app.load_config()
                self.assertIs(app.config, self.config)
                self.assertEqual(app.config_path, self.config_path)

    def test_trace_defaults_disabled_and_prompt_suggests_jsonl(self):
        app = self.app(FakeUI(texts=[None]))
        self.assertIsNone(app.trace_path)
        app.show_settings()
        self.assert_message(app.ui, "Трассировка: выключена")
        app.set_trace()
        self.assertEqual(app.ui.text_calls[0][1], "wire_menu.jsonl")
        self.assertIsNone(app.trace_path)

    def test_trace_enable_cancel_disable_does_not_touch_file(self):
        trace = self.directory / "wire.JSONL"
        trace.write_text("existing trace\n", encoding="utf-8")
        ui = FakeUI(texts=["  " + str(trace) + "  ", None, "   "])
        app = self.app(ui)
        app.set_trace()
        self.assertEqual(app.trace_path, trace)
        app.set_trace()
        self.assertEqual(app.trace_path, trace)
        self.assertEqual(ui.text_calls[1][1], str(trace))
        app.set_trace()
        self.assertIsNone(app.trace_path)
        self.assertEqual(trace.read_text(encoding="utf-8"), "existing trace\n")
        ui.assert_consumed(self)

    def test_trace_rejects_extension_or_nonexistent_parent(self):
        old_trace = self.directory / "old.jsonl"
        for path in (self.directory / "wire.csv", self.directory / "missing" / "wire.jsonl"):
            with self.subTest(path=path):
                app = self.app(FakeUI(texts=[str(path)]), trace_path=old_trace)
                app.set_trace()
                self.assertEqual(app.trace_path, old_trace)
                self.assertFalse(path.exists())
                self.assert_message(app.ui, "существующей директории")

    def test_trace_blocked_while_connected(self):
        trace = self.directory / "wire.jsonl"
        app = self.app(trace_path=trace)
        app.device = DeviceMock()
        app.set_trace()
        self.assertEqual(app.trace_path, trace)
        self.assertEqual(app.ui.text_calls, [])
        self.assert_message(app.ui, "Сначала отключите")

    def test_exit_selection_ctrl_c_and_eof_close_active_session(self):
        for answer, code in ((None, 0), (KeyboardInterrupt(), 130), (EOFError(), 0)):
            with self.subTest(answer=answer):
                ui = FakeUI(["connection", "demo", None, answer])
                device = DeviceMock()
                app = self.app(ui, DeviceFactory(device))
                self.assertEqual(app.run(), code)
                self.assertEqual(device.calls, ["start", "close"])
                self.assertTrue(device.resources_closed)
                self.assertIsNone(app.device)
                self.assertFalse(app.demo)
                self.assertEqual(self.output.getvalue(), "")
                ui.assert_consumed(self)

    def test_ctrl_c_and_eof_in_submenu_or_sample_close_session(self):
        for error, code in ((KeyboardInterrupt(), 130), (EOFError(), 0)):
            for at_sample in (False, True):
                with self.subTest(error=error, at_sample=at_sample):
                    selections = ["connection", "demo", None, "measurements"]
                    selections += ["sample"] if at_sample else [error]
                    device = DeviceMock(samples=[error] if at_sample else [])
                    app = self.app(FakeUI(selections), DeviceFactory(device))
                    self.assertEqual(app.run(), code)
                    self.assertTrue(device.resources_closed)
                    self.assertEqual(device.calls.count("close"), 1)
                    self.assertEqual(self.output.getvalue(), "")

    def test_close_error_worker_ended_sets_nonzero_but_allows_new_selection(self):
        first = DeviceMock(close_error=DeviceError("STOP not acknowledged"))
        second = DeviceMock()
        factory = DeviceFactory(first, second)
        ui = FakeUI(["connection", "demo", "disconnect", "demo", None, None])
        app = self.app(ui, factory)
        self.assertEqual(app.run(), 1)
        self.assertEqual(len(factory.calls), 2)
        self.assertTrue(first.resources_closed)
        self.assertTrue(second.resources_closed)
        self.assert_message(ui, "Ошибка завершения: STOP not acknowledged")
        self.assertIsNone(app.device)
        ui.assert_consumed(self)

    def test_close_error_alive_worker_blocks_all_changes_and_second_session(self):
        first = DeviceMock(close_error=DeviceError("worker still owns port"),
                           running_after_close=True)
        second = DeviceMock()
        factory = DeviceFactory(first, second)
        ui = FakeUI(confirmations=[True])
        app = self.app(ui, factory)
        app.connect(demo=True)
        app.disconnect()
        self.assertIs(app.device, first)
        self.assertTrue(app.demo)
        self.assertTrue(first.is_running)
        self.assertEqual(app.exit_code, 1)
        app.connect()
        app.connect(demo=True)
        app.choose_port()
        app.load_config()
        app.set_trace()
        self.assertEqual(len(factory.calls), 1)
        self.assertEqual(ui.confirm_calls, [])
        self.assertEqual(ui.select_calls, [])
        self.assertEqual(ui.text_calls, [])
        self.assert_message(ui, "Новое подключение заблокировано")
        # is_running, not the remembered close error, gates release of ownership.
        first.running_after_close = False
        app.disconnect()
        self.assertIsNone(app.device)
        app.connect()
        self.assertIs(app.device, second)
        self.assertEqual(len(factory.calls), 2)
        app.disconnect()
        ui.assert_consumed(self)

    def test_start_failure_closes_resources_and_allows_new_menu_selection(self):
        first = DeviceMock(start_error=DeviceError("start failed"),
                           close_error=DeviceError("remembered start failure"))
        second = DeviceMock()
        factory = DeviceFactory(first, second)
        ui = FakeUI(["connection", "demo", "demo", None, None])
        app = self.app(ui, factory)
        self.assertEqual(app.run(), 1)
        self.assertEqual(first.calls, ["start", "close"])
        self.assertTrue(first.resources_closed)
        self.assertEqual(second.calls, ["start", "close"])
        self.assertEqual(len(factory.calls), 2)
        self.assert_message(ui, "Ошибка: start failed")
        self.assertIsNone(app.device)
        ui.assert_consumed(self)

    def test_start_failure_alive_worker_prevents_next_demo_until_cleanup(self):
        device = DeviceMock(start_error=DeviceError("start failed"),
                            close_error=DeviceError("join timed out"), running_after_close=True)
        factory = DeviceFactory(device)
        app = self.app(factory=factory)
        with self.assertRaisesRegex(DeviceError, "start failed"):
            app.connect(demo=True)
        self.assertIs(app.device, device)
        app.connect(demo=True)
        self.assertEqual(len(factory.calls), 1)
        self.assertFalse(device.resources_closed)
        device.running_after_close = False
        app.disconnect()
        self.assertTrue(device.resources_closed)
        self.assertIsNone(app.device)

    def test_keyboard_interrupt_during_start_closes_resources(self):
        device = DeviceMock(start_error=KeyboardInterrupt())
        app = self.app(FakeUI(["connection", "demo"]), DeviceFactory(device))
        self.assertEqual(app.run(), 130)
        self.assertEqual(device.calls, ["start", "close"])
        self.assertTrue(device.resources_closed)

    def test_acquire_failure_disconnects_reports_nonzero_and_no_csv(self):
        device = DeviceMock(samples=[DeviceError("receiver failed")],
                            close_error=DeviceError("receiver failed"))
        ui = FakeUI(["connection", "demo", None, "measurements", "sample", None, None])
        app = self.app(ui, DeviceFactory(device))
        self.assertEqual(app.run(), 1)
        self.assertEqual(device.calls, ["start", "sample", "close"])
        self.assertIsNone(app.device)
        self.assertEqual(self.output.getvalue(), "")
        self.assert_message(ui, "Ошибка: receiver failed")

    def test_background_failure_discovered_on_exit_returns_nonzero(self):
        device = DeviceMock(close_error=DeviceError("background receive failure"))
        ui = FakeUI(["connection", "demo", None, None])
        app = self.app(ui, DeviceFactory(device))
        self.assertEqual(app.run(), 1)
        self.assertTrue(device.resources_closed)
        self.assert_message(ui, "Ошибка завершения: background receive failure")
        self.assertEqual(self.output.getvalue(), "")

    def test_main_closes_ui_and_passes_port_override_without_mutating_json(self):
        ui = FakeUI()
        with patch("com_reader.menu.QuestionaryUI", return_value=ui), \
                patch("com_reader.menu.MenuApplication") as application, \
                patch("sys.stdin.isatty", return_value=True), \
                patch("sys.stderr.isatty", return_value=True):
            application.return_value.run.return_value = 0
            self.assertEqual(main(["--config", str(self.config_path), "--port", " COM88 "]), 0)
        args, kwargs = application.call_args
        self.assertEqual(args[0].serial.port, "COM88")
        self.assertIs(args[1], ui)
        self.assertEqual(kwargs["config_path"], self.config_path)
        self.assertIsNone(kwargs["trace_path"])
        self.assertTrue(ui.closed)
        self.assertEqual(self.config_path.read_bytes(), self.original_json)

    def test_main_invalid_config_port_trace_or_noninteractive_never_constructs_ui(self):
        cases = [
            ["--config", str(self.directory / "missing.json")],
            ["--config", str(self.config_path), "--port", "   "],
            ["--config", str(self.config_path), "--trace", str(self.directory / "bad.csv")],
            ["--config", str(self.config_path), "--trace",
             str(self.directory / "missing" / "wire.jsonl")],
            ["--config", str(self.config_path)],
        ]
        for argv in cases:
            with self.subTest(argv=argv), patch("com_reader.menu.QuestionaryUI") as ui, \
                    patch("com_reader.menu.MenuApplication") as application, \
                    patch("sys.stdin.isatty", return_value=False), \
                    patch("sys.stderr.isatty", return_value=False), \
                    patch("sys.stderr", io.StringIO()) as stderr:
                self.assertEqual(main(argv), 2)
                ui.assert_not_called()
                application.assert_not_called()
                self.assertIn("Ошибка меню:", stderr.getvalue())

    def test_main_closes_ui_even_when_run_is_interrupted(self):
        ui = FakeUI()
        with patch("com_reader.menu.QuestionaryUI", return_value=ui), \
                patch("com_reader.menu.MenuApplication") as application, \
                patch("sys.stdin.isatty", return_value=True), \
                patch("sys.stderr.isatty", return_value=True):
            application.return_value.run.side_effect = KeyboardInterrupt()
            self.assertEqual(main(["--config", str(self.config_path)]), 130)
        self.assertTrue(ui.closed)


class SyntheticDeviceTests(MenuTestCase):
    def setUp(self):
        self.hardware = self.enterContext(patch(
            "com_reader.device.open_serial", side_effect=AssertionError("Real COM forbidden")))

    def cleanup_device(self, device):
        try:
            device.close()
        except DeviceError:
            pass
        self.assertFalse(device.is_running, "Synthetic owner worker leaked")

    def test_public_is_running_tracks_owner_even_before_ready_and_after_stop(self):
        entered, release = threading.Event(), threading.Event()
        config = Config.from_dict(config_data())
        transport = SimulatedSerial(0.01)

        def factory(*args):
            entered.set()
            if not release.wait(WAIT_SECONDS):
                raise AssertionError("Factory release timed out")
            return transport

        device = Device(config, serial_factory=factory)
        self.assertFalse(device.is_running)
        errors = []

        def start():
            try:
                device.start()
            except BaseException as exc:
                errors.append(exc)

        caller = threading.Thread(target=start, daemon=True)

        def cleanup():
            release.set()
            caller.join(WAIT_SECONDS)
            self.cleanup_device(device)
            self.assertFalse(caller.is_alive(), "start caller leaked")

        self.addCleanup(cleanup)
        caller.start()
        self.assertTrue(entered.wait(WAIT_SECONDS))
        self.assertTrue(device.is_running)
        self.assertFalse(device._ready)
        device._stop.set()
        self.assertTrue(device.is_running, "stop request must not hide live ownership")
        release.set()
        caller.join(WAIT_SECONDS)
        self.assertFalse(caller.is_alive())
        with self.assertRaises(DeviceError):
            device.close()
        self.assertFalse(device.is_running)
        self.assertFalse(transport.is_open)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], DeviceError)

    def test_actual_demo_device_produces_csv_and_closes_without_hardware(self):
        ui = FakeUI(["connection", "demo", None, "measurements", "sample", None, None])
        output = io.StringIO()
        devices = []

        def factory(config, **kwargs):
            device = Device(config, **kwargs)
            devices.append(device)
            self.addCleanup(self.cleanup_device, device)
            return device

        app = MenuApplication(Config.from_dict(config_data()), ui, output=output,
                              device_factory=factory, ports_provider=lambda: [])
        self.assertEqual(app.run(), 0)
        rows = list(csv.reader(io.StringIO(output.getvalue())))
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]), 3)
        counter = int(rows[0][1]) - 200
        self.assertEqual(int(rows[0][2]), 500 + counter)
        self.assertFalse(devices[0].is_running)
        self.assertFalse(devices[0]._port.is_open)
        self.hardware.assert_not_called()

    def test_actual_background_failure_idle_exit_returns_nonzero_and_closes(self):
        failed = threading.Event()

        class FailingSerial(SimulatedSerial):
            def __init__(self):
                super().__init__(0.01)
                self.fail = threading.Event()
                self.commands = []

            def read(self, size):
                if self.fail.is_set():
                    self.fail.clear()
                    raise OSError("synthetic background read failed")
                return super().read(size)

            def write(self, data):
                self.commands.append(data)
                return super().write(data)

        transport = FailingSerial()
        devices = []
        ui = FakeUI()
        output = io.StringIO()

        def factory(config, **kwargs):
            def log(message):
                ui.report(message)
                if "Приём завершён" in message:
                    failed.set()

            device = Device(config, log=log, trace_path=kwargs["trace_path"],
                            serial_factory=lambda *args: transport)
            devices.append(device)
            self.addCleanup(self.cleanup_device, device)
            return device

        def quit_after_failure():
            transport.fail.set()
            self.assertTrue(failed.wait(WAIT_SECONDS), "Worker did not report failure")
            return None

        ui.selections.extend(["connection", "demo", None, quit_after_failure])
        app = MenuApplication(Config.from_dict(config_data()), ui, output=output,
                              device_factory=factory, ports_provider=lambda: [])
        self.assertEqual(app.run(), 1)
        self.assertFalse(transport.is_open)
        self.assertFalse(devices[0].is_running)
        self.assertEqual(transport.commands.count(STOP_COMMAND), 2)
        self.assertEqual(output.getvalue(), "")
        self.assertTrue(any("Ошибка завершения" in m and "synthetic background read failed" in m
                            for m in ui.messages), ui.messages)
        self.hardware.assert_not_called()


@unittest.skipUnless(HAS_QUESTIONARY, "optional questionary is not installed")
class QuestionaryUITests(MenuTestCase):
    def setUp(self):
        # No questionary/prompt_toolkit import is required by the stdlib-only tests.
        import questionary
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput

        self.questionary = questionary
        self.contexts = ExitStack()
        self.addCleanup(self.contexts.close)
        self.pipe = self.contexts.enter_context(create_pipe_input())
        self.stdout, self.stderr = io.StringIO(), io.StringIO()
        self.contexts.enter_context(patch("sys.stdout", self.stdout))
        self.contexts.enter_context(patch("sys.stderr", self.stderr))
        stderr = self.stderr

        class CapturingOutput(DummyOutput):
            def write(self, data):
                stderr.write(data)

            def write_raw(self, data):
                stderr.write(data)

        self.output = CapturingOutput()
        # Isolate prompt behavior from the constructor's Input/context-manager bug.
        # Forward Input methods too, so these tests also work after the fix.
        # The raw Input API is exercised separately below, without this adapter.
        pipe = self.pipe

        class ManagedInput:
            def __enter__(self):
                return pipe

            def __exit__(self, *args):
                pass

            def __getattr__(self, name):
                return getattr(pipe, name)

        self.create_input = self.contexts.enter_context(patch(
            "prompt_toolkit.input.defaults.create_input", return_value=ManagedInput()))
        self.create_output = self.contexts.enter_context(patch(
            "prompt_toolkit.output.defaults.create_output", return_value=self.output))
        # Use the real library applications with a watchdog, never an interactive TTY.
        for name in ("select", "text", "confirm"):
            real_factory = getattr(questionary, name)

            def bounded_factory(*args, _factory=real_factory, **kwargs):
                question = _factory(*args, **kwargs)
                application = question.application

                async def watchdog():
                    await asyncio.sleep(WAIT_SECONDS)
                    application.exit(exception=AssertionError("Prompt exceeded bounded wait"))

                application.pre_run_callables.append(
                    lambda: application.create_background_task(watchdog()))
                return question

            self.contexts.enter_context(patch.object(questionary, name, side_effect=bounded_factory))
        self.ui = QuestionaryUI()
        self.contexts.callback(self.ui.close)
        self.hardware = self.contexts.enter_context(patch(
            "com_reader.device.open_serial", side_effect=AssertionError("Real COM forbidden")))

    def test_constructor_accepts_actual_input_api_without_context_manager(self):
        from prompt_toolkit.input import Input

        self.assertIsInstance(self.pipe, Input)
        self.create_input.return_value = self.pipe
        try:
            ui = QuestionaryUI()
        except TypeError as exc:
            self.fail(f"create_input returns Input, not a context manager: {exc}")
        else:
            ui.close()

    def test_real_library_none_value_becomes_label_but_ui_maps_sentinel_to_back(self):
        self.assertEqual(self.questionary.Choice("Back", value=None).value, "Back")
        self.pipe.send_text("\r")
        self.assertIsNone(self.ui.select("Choose", [("Back", None), ("Connect", "connect")]))

    def test_arrow_down_enter_returns_value_not_label(self):
        self.pipe.send_text("\x1b[B\r")
        self.assertEqual(self.ui.select("Choose", [("First", "first"), ("Second", "second")]),
                         "second")

    def test_arrow_up_wrap_and_enter_returns_back_none(self):
        self.pipe.send_text("\x1b[A\r")
        self.assertIsNone(self.ui.select("Choose", [("Connect", "connect"), ("Back", None)]))

    def test_default_confirmation_enter_is_false(self):
        self.pipe.send_text("\r")
        self.assertIs(self.ui.confirm("Send STOP and START?"), False)
        self.questionary.confirm.assert_called_once()
        self.assertIs(self.questionary.confirm.call_args.kwargs["default"], False)

    def test_confirm_explicit_yes_is_true(self):
        self.pipe.send_text("y\r")
        self.assertIs(self.ui.confirm("Proceed?"), True)

    def test_text_uses_real_prompt_and_initial_default(self):
        self.pipe.send_text("\r")
        self.assertEqual(self.ui.text("Port", "COM42"), "COM42")
        self.assertEqual(self.questionary.text.call_args.kwargs["default"], "COM42")

    def test_ctrl_c_select_text_and_confirm_propagates_for_cleanup(self):
        for prompt in (lambda: self.ui.select("Choose", [("Back", None)]),
                       lambda: self.ui.text("Port"), lambda: self.ui.confirm("Proceed?")):
            with self.subTest(prompt=prompt):
                self.pipe.send_text("\x03")
                with self.assertRaises(KeyboardInterrupt):
                    prompt()

    def test_closed_input_raises_eof(self):
        self.pipe.close()
        with self.assertRaises(EOFError):
            self.ui.select("Choose", [("Back", None)])

    def test_prompts_and_report_go_to_stderr_not_csv_stdout(self):
        self.create_output.assert_called_once_with(stdout=self.stderr)
        self.create_input.assert_called_once()
        self.pipe.send_text("\r")
        self.ui.select("UI-PROMPT-MARKER", [("Back", None)])
        self.ui.report("UI-STATUS-MARKER")
        # StdoutProxy drains its queued writes on close.
        self.ui.close()
        self.assertEqual(self.stdout.getvalue(), "")
        self.assertIn("UI-PROMPT-MARKER", self.stderr.getvalue())
        self.assertIn("UI-STATUS-MARKER", self.stderr.getvalue())

    def test_real_ui_root_exit_ctrl_c_and_eof_clean_up_session(self):
        for action, code in (("exit", 0), ("interrupt", 130), ("eof", 0)):
            with self.subTest(action=action):
                device = DeviceMock()
                app = MenuApplication(Config.from_dict(config_data()), self.ui,
                                      output=self.stdout, device_factory=DeviceFactory(device),
                                      ports_provider=lambda: [])
                app.connect(demo=True)
                if action == "exit":
                    self.pipe.send_text("\x1b[A\r")
                elif action == "interrupt":
                    self.pipe.send_text("\x03")
                else:
                    self.pipe.close()
                self.assertEqual(app.run(), code)
                self.assertEqual(device.calls, ["start", "close"])
                self.assertTrue(device.resources_closed)
                self.assertIsNone(app.device)
        self.ui.close()
        self.assertEqual(self.stdout.getvalue(), "")

    def test_q_exits_cleanly_instead_of_requiring_followup_ctrl_c(self):
        device = DeviceMock()
        app = MenuApplication(Config.from_dict(config_data()), self.ui,
                              output=self.stdout, device_factory=DeviceFactory(device),
                              ports_provider=lambda: [])
        app.connect(demo=True)
        # Ctrl+C is a bounded fallback: a working q binding exits before consuming it.
        self.pipe.send_text("q\x03")
        result = app.run()
        self.assertTrue(device.resources_closed)
        self.assertEqual(device.calls, ["start", "close"])
        self.assertEqual(result, 0, "q is ignored; the following Ctrl+C caused exit code 130")


if __name__ == "__main__":
    unittest.main()
