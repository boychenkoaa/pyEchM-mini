"""Failure-path regressions using only in-memory serial I/O and bounded waits."""

from collections import deque
from dataclasses import replace
import io
import json
from pathlib import Path
import threading
import unittest
from unittest.mock import Mock, patch

from com_reader.cli import main
from com_reader.config import Config
from com_reader.device import CommandError, Device, DeviceError
from com_reader.protocol import START_COMMAND, STOP_COMMAND


OLD_FRAME = b":1000001000210"
NEW_FRAME = b":1000003000410"
WAIT_SECONDS = 2


def settings():
    return Config.from_dict({
        "schema_version": 1,
        "serial": {"port": "TEST-NO-HARDWARE"},
        "measurement": {
            "mode": "single", "channels": [2, 5],
            "channel_delays": [1] * 32, "interval_seconds": 1, "filter_hz": 500,
        },
        "protocol": {
            "read_timeout_seconds": 0.01,
            "command_timeout_seconds": 0.5,
            "sample_timeout_seconds": 3,
        },
    })


class ScriptedSerial:
    """Preserve injected chunk boundaries; consume injected exceptions only once."""

    def __init__(self, *, start_reply=b":B\n", filter_chunks=(b":F\n",)):
        self.condition = threading.Condition()
        self.chunks = deque()
        self.start_reply = start_reply
        self.filter_chunks = filter_chunks
        self.commands = []
        self.read_chunks = []
        self.stop_count = 0
        self.closed = threading.Event()

    def inject(self, *chunks):
        with self.condition:
            self.chunks.extend(chunks)
            self.condition.notify_all()

    @property
    def in_waiting(self):
        with self.condition:
            if self.chunks and isinstance(self.chunks[0], bytes):
                return len(self.chunks[0])
            return 0

    def read(self, size):
        with self.condition:
            self.condition.wait_for(lambda: self.chunks or self.closed.is_set(), 0.01)
            if not self.chunks:
                return b""
            chunk = self.chunks.popleft()
            if isinstance(chunk, Exception):
                raise chunk
            result = chunk[:size]
            if chunk[size:]:
                self.chunks.appendleft(chunk[size:])
            self.read_chunks.append(result)
            return result

    def write(self, command):
        self.commands.append(command)
        if command == STOP_COMMAND:
            self.stop_count += 1
            self.inject(b":C\n")
        elif command.startswith(b":5"):
            self.inject(*self.filter_chunks)
        elif command.startswith(b":2"):
            self.inject(b":A\n")
        elif command == START_COMMAND:
            self.inject(self.start_reply)
        else:
            raise AssertionError(f"Unexpected command: {command!r}")
        return len(command)

    def reset_input_buffer(self):
        with self.condition:
            self.chunks.clear()

    def close(self):
        with self.condition:
            self.closed.set()
            self.condition.notify_all()


class FailingTrace(io.StringIO):
    def __init__(self, operation, target):
        super().__init__()
        self.operation = operation
        self.target = target
        self.failure = OSError(28, "trace disk full")
        self.failures = 0
        self.attempts_after_failure = 0
        self.stop_records = 0
        self.fail_this_record = False
        self.armed = threading.Event()

    def write(self, text):
        if self.failures:
            self.attempts_after_failure += 1
        record = json.loads(text)
        if record["direction"] == "tx" and record["hex"] == STOP_COMMAND.hex():
            self.stop_records += 1
        self.fail_this_record = (
            record["direction"] == "rx" and self.armed.is_set()
            if self.target == "runtime"
            else self.stop_records == 2 and record["direction"] == "tx"
        )
        if self.operation == "write" and self.fail_this_record:
            self.failures += 1
            raise self.failure
        return super().write(text)

    def flush(self):
        if self.operation == "flush" and self.fail_this_record:
            self.failures += 1
            raise self.failure
        return super().flush()


class FakeClock:
    def __init__(self):
        self.milliseconds = 0

    def advance(self, milliseconds):
        self.milliseconds += milliseconds

    def monotonic(self):
        return self.milliseconds / 1000

    def monotonic_ns(self):
        return self.milliseconds * 1_000_000


class TimedSerial:
    def __init__(self, clock, write_ms, *, reply=b""):
        self.clock = clock
        self.write_ms = write_ms
        self.reply = reply
        self.commands = []
        self.read_count = 0

    @property
    def in_waiting(self):
        return len(self.reply)

    def reset_input_buffer(self):
        pass

    def write(self, command):
        self.commands.append(command)
        self.clock.advance(self.write_ms)
        return len(command)

    def read(self, size):
        self.read_count += 1
        self.clock.advance(100)
        result, self.reply = self.reply[:size], self.reply[size:]
        return result

    def close(self):
        self.reply = b""


class FailurePathTests(unittest.TestCase):
    def cleanup_device(self, device):
        try:
            device.close()
        except DeviceError:
            # A remembered failure is now part of the public close() contract.
            pass
        if device._thread is not None:
            self.assertFalse(device._thread.is_alive(), "Serial worker leaked")

    def make_device(self, transport=None, **kwargs):
        transport = ScriptedSerial() if transport is None else transport
        messages = []
        device = Device(settings(), serial_factory=lambda *args: transport,
                        log=messages.append, **kwargs)
        self.addCleanup(self.cleanup_device, device)
        return device, transport, messages

    def check_trace_failure(self, operation, target):
        trace = FailingTrace(operation, target)
        self.addCleanup(trace.close)
        device, transport, messages = self.make_device(trace_path="unused-wire.jsonl")
        with patch.object(Path, "open", return_value=trace) as open_trace:
            device.start()
        open_trace.assert_called_once_with("a", encoding="utf-8")
        if target == "runtime":
            # An idle read(size=1) may split the frame; fail on the first RX chunk.
            trace.armed.set()
            transport.inject(OLD_FRAME)
            self.assertTrue(transport.closed.wait(WAIT_SECONDS))
        with self.assertRaisesRegex(DeviceError, "trace disk full"):
            device.close()
        self.assertTrue(transport.closed.is_set())
        self.assertEqual(transport.stop_count, 2)
        self.assertEqual(transport.commands[-1], STOP_COMMAND)
        self.assertTrue(trace.closed)
        self.assertEqual(trace.failures, 1)
        self.assertEqual(trace.attempts_after_failure, 0)
        self.assertTrue(any("подтвердил остановку" in message for message in messages))
        if target == "runtime":
            self.assertTrue(any("Ошибка фонового приёма" in message for message in messages))
        else:
            self.assertTrue(any("аварийный STOP" in message for message in messages))

    def test_runtime_trace_write_failure_still_sends_stop_and_closes(self):
        self.check_trace_failure("write", "runtime")

    def test_runtime_trace_flush_failure_still_sends_stop_and_closes(self):
        self.check_trace_failure("flush", "runtime")

    def test_first_trace_write_failure_on_final_stop_still_sends_and_closes(self):
        self.check_trace_failure("write", "shutdown")

    def test_first_trace_flush_failure_on_final_stop_still_sends_and_closes(self):
        self.check_trace_failure("flush", "shutdown")

    def test_filter_ack_after_data_in_separate_chunks(self):
        transport = ScriptedSerial(filter_chunks=(OLD_FRAME, b":", b"F\n"))
        device, transport, messages = self.make_device(transport)
        device.start()
        device.close()
        self.assertIn(START_COMMAND, transport.commands)
        self.assertEqual(transport.stop_count, 2)
        self.assertTrue(transport.closed.is_set())
        self.assertIn(OLD_FRAME, transport.read_chunks)
        self.assertIn(b":", transport.read_chunks)
        self.assertIn(b"F\n", transport.read_chunks)
        reply_hex = (OLD_FRAME + b":F\n").hex()
        self.assertTrue(any(reply_hex in message for message in messages))

    def test_command_deadline_includes_write_time(self):
        clock = FakeClock()
        config = settings()
        config = replace(config, protocol=replace(
            config.protocol, command_timeout_seconds=1, read_timeout_seconds=0.1))
        transport = TimedSerial(clock, write_ms=750)
        device = Device(config)
        device._port = transport
        with patch("com_reader.device.time", clock):
            with self.assertRaises(CommandError):
                device._exchange(START_COMMAND, b"B", "test START")
        self.assertEqual(transport.commands, [START_COMMAND])
        self.assertEqual(transport.read_count, 3)
        self.assertEqual(clock.milliseconds, 1050)

    def test_write_exhausting_deadline_does_not_accept_buffered_ack(self):
        clock = FakeClock()
        config = settings()
        config = replace(config, protocol=replace(config.protocol, command_timeout_seconds=1))
        transport = TimedSerial(clock, write_ms=1000, reply=b":B\n")
        device = Device(config)
        device._port = transport
        with patch("com_reader.device.time", clock):
            with self.assertRaises(CommandError):
                device._exchange(START_COMMAND, b"B", "test START")
        self.assertEqual(transport.read_count, 0)
        self.assertEqual(transport.commands, [START_COMMAND])

    def test_close_budget_covers_interrupted_exchange_and_final_stop(self):
        config = settings()
        config = replace(config, protocol=replace(
            config.protocol, command_timeout_seconds=3, read_timeout_seconds=0.1))
        device = Device(config)
        clock = FakeClock()
        worker = Mock()
        worker.is_alive.return_value = True

        def join(timeout):
            # Two exchanges plus bounded read overshoot take 6.2 virtual seconds.
            elapsed_ms = min(round(timeout * 1000), 6200)
            clock.advance(elapsed_ms)
            worker.is_alive.return_value = elapsed_ms < 6200

        worker.join.side_effect = join
        device._thread = worker
        device.close()
        worker.join.assert_called_once()
        self.assertAlmostEqual(worker.join.call_args.args[0], 8.2)
        self.assertEqual(clock.milliseconds, 6200)
        self.assertTrue(device._stop.is_set())
        self.assertFalse(worker.is_alive())

    def test_one_read_error_is_reported_even_when_stop_succeeds(self):
        device, transport, messages = self.make_device()
        device.start()
        failure = OSError("receive failed once")
        transport.inject(failure)
        self.assertTrue(transport.closed.wait(WAIT_SECONDS))
        with self.assertRaisesRegex(DeviceError, "receive failed once") as raised:
            device.close()
        self.assertIs(raised.exception.__cause__, failure)
        self.assertEqual(transport.stop_count, 2)
        self.assertEqual(transport.commands[-1], STOP_COMMAND)
        self.assertTrue(any("Ошибка фонового приёма" in message for message in messages))
        self.assertTrue(any("подтвердил остановку" in message for message in messages))

    def test_cli_q_after_worker_failure_returns_one_without_csv(self):
        device, transport, messages = self.make_device()
        test = self

        class QuitAfterFailure:
            def readline(self):
                transport.inject(OSError("receive failed while idle"))
                test.assertTrue(transport.closed.wait(WAIT_SECONDS))
                return "q\n"

        output = io.StringIO()
        with patch("com_reader.cli.Config.load", return_value=device.config), \
                patch("com_reader.cli.Device", return_value=device) as factory, \
                patch("com_reader.cli._report", side_effect=messages.append), \
                patch("sys.stdin", QuitAfterFailure()), \
                patch("sys.stdout", output), patch("sys.stderr", io.StringIO()):
            result = main([])
        factory.assert_called_once()
        self.assertEqual(result, 1)
        self.assertEqual(output.getvalue(), "")
        self.assertTrue(transport.closed.is_set())
        self.assertEqual(transport.stop_count, 2)
        self.assertTrue(any("Ошибка завершения" in message and "receive failed while idle" in message
                            for message in messages))

    def test_start_ack_partial_frame_is_preserved_but_excluded_after_request(self):
        prefix = OLD_FRAME[:8]
        device, transport, _ = self.make_device(ScriptedSerial(start_reply=b":B\n" + prefix))
        partial_seen = threading.Event()
        request_waiting = threading.Event()
        parsed_frames = []
        result = []
        original_feed = device.parser.feed
        original_wait = device._condition.wait

        def feed(data, received_ns):
            frames = original_feed(data, received_ns)
            parsed_frames.extend(frames)
            if device.parser.buffered_bytes == len(prefix):
                partial_seen.set()
            return frames

        def wait(timeout=None):
            request_waiting.set()
            return original_wait(timeout)

        def request():
            try:
                result.append(device.wait_next_sample())
            except Exception as exc:
                result.append(exc)

        worker = threading.Thread(target=request, daemon=True)

        def cleanup_request():
            self.cleanup_device(device)
            if worker.ident is not None:
                worker.join(WAIT_SECONDS)
                self.assertFalse(worker.is_alive(), "Sample waiter leaked")

        self.addCleanup(cleanup_request)
        with patch.object(device.parser, "feed", side_effect=feed), \
                patch.object(device._condition, "wait", side_effect=wait):
            device.start()
            self.assertTrue(partial_seen.wait(WAIT_SECONDS))
            worker.start()
            self.assertTrue(request_waiting.wait(WAIT_SECONDS))
            transport.inject(OLD_FRAME[len(prefix):] + NEW_FRAME)
            worker.join(WAIT_SECONDS)
            self.assertFalse(worker.is_alive())
            device.close()
        self.assertEqual(len(result), 1)
        self.assertNotIsInstance(result[0], Exception)
        self.assertEqual(result[0].values, (3, 4))
        self.assertEqual(result[0].sequence, 2)
        self.assertEqual([frame.raw for frame in parsed_frames], [OLD_FRAME, NEW_FRAME])


if __name__ == "__main__":
    unittest.main()
