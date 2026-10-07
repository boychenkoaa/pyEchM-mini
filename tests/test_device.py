import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from com_reader.config import Config
from com_reader.device import Device, DeviceError, SampleTimeout
from com_reader.protocol import configure_command


def settings():
    return Config.from_dict({
        "schema_version": 1,
        "serial": {"port": "TEST"},
        "measurement": {"mode": "single", "channels": [2, 5], "channel_delays": [1] * 32,
                        "interval_seconds": 1, "filter_hz": 500},
        "protocol": {"read_timeout_seconds": 0.005, "command_timeout_seconds": 0.05,
                     "sample_timeout_seconds": 0.3},
    })


class ManualSerial:
    def __init__(self, *, start_reply=b":B\n", stop_reply=b":C\n", short_write=False):
        self.buffer = bytearray()
        self.condition = threading.Condition()
        self.closed = False
        self.error = None
        self.commands = []
        self.thread_ids = set()
        self.start_reply = start_reply
        self.stop_reply = stop_reply
        self.short_write = short_write
        self.stop_count = 0

    def inject(self, data):
        with self.condition:
            self.buffer.extend(data)
            self.condition.notify_all()

    @property
    def in_waiting(self):
        with self.condition:
            return len(self.buffer)

    def read(self, size):
        self.thread_ids.add(threading.get_ident())
        with self.condition:
            if not self.buffer and self.error is None:
                self.condition.wait(0.005)
            if self.error:
                raise self.error
            result = bytes(self.buffer[:size])
            del self.buffer[:size]
            return result

    def write(self, command):
        self.thread_ids.add(threading.get_ident())
        self.commands.append(command)
        if command == b":4\n":
            self.stop_count += 1
            self.inject(b":C\n" if self.stop_count == 1 else self.stop_reply)
        elif command.startswith(b":5"):
            self.inject(b":F\n")  # Filter ACK deliberately differs from the simulator.
        elif command.startswith(b":2"):
            if self.short_write:
                return len(command) - 1
            self.inject(b":A\n")
        elif command == b":3\n":
            self.inject(self.start_reply)
        return len(command)

    def reset_input_buffer(self):
        self.thread_ids.add(threading.get_ident())
        with self.condition:
            self.buffer.clear()

    def fail(self):
        with self.condition:
            self.error = OSError("Disconnected")
            self.condition.notify_all()

    def close(self):
        self.thread_ids.add(threading.get_ident())
        self.closed = True


def until(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for test state")
        time.sleep(0.001)


class DeviceTests(unittest.TestCase):
    def make_device(self, transport=None, **kwargs):
        transport = ManualSerial() if transport is None else transport
        device = Device(settings(), serial_factory=lambda *args: transport, **kwargs)

        def cleanup():
            try:
                device.close()
            except DeviceError:
                pass

        self.addCleanup(cleanup)
        return device, transport

    def request(self, device):
        result = []

        def wait():
            try:
                result.append(device.wait_next_sample())
            except DeviceError as exc:
                result.append(exc)

        worker = threading.Thread(target=wait, daemon=True)
        worker.start()
        self.addCleanup(worker.join, 1)
        until(lambda: device._request_ns is not None)
        return worker, result

    def test_commands_and_all_io_owned_by_one_worker(self):
        device, transport = self.make_device()
        device.start()
        device.close()
        self.assertEqual(transport.commands, [b":4\n", b":5\x03", configure_command(settings().measurement), b":3\n", b":4\n"])
        self.assertTrue(transport.closed)
        self.assertEqual(len(transport.thread_ids), 1)
        self.assertNotIn(threading.get_ident(), transport.thread_ids)

    def test_enter_returns_first_new_frame_not_latest_or_cached(self):
        device, transport = self.make_device()
        device.start()
        transport.inject(b":1000001000210")
        until(lambda: device._sequence == 1)
        worker, result = self.request(device)
        transport.inject(b":1000003000410:1000005000610")
        worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertEqual(result[0].values, (3, 4))
        self.assertEqual(result[0].channels, (2, 5))
        with self.assertRaises(SampleTimeout):
            device.wait_next_sample(0.015)

    def test_partial_pretrigger_frame_is_excluded(self):
        device, transport = self.make_device()
        device.start()
        transport.inject(b":1000001")
        until(lambda: device.parser.buffered_bytes == 8)
        worker, result = self.request(device)
        transport.inject(b"000210:100FFFF800010")
        worker.join(1)
        self.assertEqual(result[0].values, (-1, -32768))
        self.assertEqual(result[0].sequence, 2)

    def test_data_with_start_ack_is_preserved(self):
        device, _ = self.make_device(ManualSerial(start_reply=b":B\n:1000001000210"))
        device.start()
        until(lambda: device._sequence == 1)
        with self.assertRaises(SampleTimeout):
            device.wait_next_sample(0.015)

    def test_timeout_without_data(self):
        device, _ = self.make_device()
        device.start()
        with self.assertRaisesRegex(SampleTimeout, "--trace"):
            device.wait_next_sample(0.015)
        self.assertIsNone(device._request_ns)

    def test_concurrent_request_rejected(self):
        device, transport = self.make_device()
        device.start()
        worker, result = self.request(device)
        with self.assertRaisesRegex(DeviceError, "Другой запрос"):
            device.wait_next_sample()
        transport.inject(b":1000001000210")
        worker.join(1)
        self.assertEqual(result[0].values, (1, 2))

    def test_disconnect_wakes_waiter_and_closes_port(self):
        device, transport = self.make_device()
        device.start()
        worker, result = self.request(device)
        transport.fail()
        worker.join(1)
        self.assertIsInstance(result[0], DeviceError)
        self.assertIn("Disconnected", str(result[0]))
        until(lambda: transport.closed)

    def test_close_cancels_waiter(self):
        device, _ = self.make_device()
        device.start()
        worker, result = self.request(device)
        device.close()
        worker.join(1)
        self.assertIsInstance(result[0], DeviceError)
        self.assertFalse(worker.is_alive())

    def test_failed_start_still_sends_stop(self):
        device, transport = self.make_device(ManualSerial(start_reply=b""))
        with self.assertRaisesRegex(DeviceError, "Запуск измерений"):
            device.start()
        with self.assertRaisesRegex(DeviceError, "Приём завершился с ошибкой"):
            device.close()
        self.assertEqual(transport.commands[-1], b":4\n")
        self.assertTrue(transport.closed)

    def test_unconfirmed_shutdown_is_reported_but_port_closed(self):
        device, transport = self.make_device(ManualSerial(stop_reply=b""))
        device.start()
        with self.assertRaisesRegex(DeviceError, "завершение не подтверждено"):
            device.close()
        self.assertTrue(transport.closed)

    def test_partial_command_write_aborts_initialization(self):
        device, transport = self.make_device(ManualSerial(short_write=True))
        with self.assertRaisesRegex(DeviceError, "Неполная запись"):
            device.start()
        with self.assertRaisesRegex(DeviceError, "Неполная запись"):
            device.close()
        self.assertNotIn(b":3\n", transport.commands)
        self.assertTrue(transport.closed)

    def test_raw_trace_contains_exact_tx_and_rx(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "wire.jsonl"
            device, transport = self.make_device(trace_path=trace)
            device.start()
            worker, result = self.request(device)
            transport.inject(b":1000001FFFF10")
            worker.join(1)
            device.close()
            records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
            transmitted = [bytes.fromhex(r["hex"]) for r in records if r["direction"] == "tx"]
            received = b"".join(bytes.fromhex(r["hex"]) for r in records if r["direction"] == "rx")
            self.assertEqual(transmitted, transport.commands)
            self.assertIn(b":1000001FFFF10", received)
            self.assertEqual(result[0].values, (1, -1))
            self.assertTrue(all(r["length"] == len(bytes.fromhex(r["hex"])) for r in records))

    def test_missing_trace_directory_never_opens_port(self):
        calls = []

        def factory(*args) -> ManualSerial:
            calls.append(args)
            raise AssertionError("Port must not open when the trace directory is missing")

        device = Device(settings(), serial_factory=factory, trace_path="does-not-exist/sub/wire.jsonl")
        with self.assertRaises(DeviceError):
            device.start()
        with self.assertRaises(DeviceError):
            device.close()
        self.assertEqual(calls, [])

    def test_invalid_lifecycle(self):
        device, _ = self.make_device()
        with self.assertRaises(DeviceError):
            device.wait_next_sample()
        device.start()
        with self.assertRaises(DeviceError):
            device.start()
        device.close()
        with self.assertRaises(DeviceError):
            device.wait_next_sample()


if __name__ == "__main__":
    unittest.main()
