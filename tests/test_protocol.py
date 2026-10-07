"""Wire-format tests: document fixtures and independent byte-level expectations."""

import hashlib
import unittest
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

from com_reader.config import MeasurementSettings
from com_reader.protocol import (
    START_COMMAND,
    STOP_COMMAND,
    FrameParser,
    configure_command,
    filter_command,
    signed_word,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TERMINATORS = (b"10", b"\r\n", b"\n")


def packets(name):
    # LF separates fixture records; it is not part of these document packets.
    return (FIXTURES / name).read_bytes().splitlines()


def expected_values(packet):
    payload = packet[4:-2]
    return tuple(
        int.from_bytes(bytes.fromhex(payload[i:i + 4].decode("ascii")),
                       byteorder="big", signed=True)
        for i in range(0, len(payload), 4)
    )


class CommandTests(unittest.TestCase):
    def test_start_stop_and_binary_frequency_commands(self):
        self.assertEqual(START_COMMAND, b":3\n")
        self.assertEqual(STOP_COMMAND, b":4\n")
        for frequency, expected in ((50, b":5\x01"), (250, b":5\x02"),
                                    (500, b":5\x03")):
            with self.subTest(frequency=frequency):
                self.assertEqual(filter_command(frequency), expected)
                self.assertEqual(len(filter_command(frequency)), 3)
        for frequency in (0, 1, 49, 100, 251, 501, True, None, "500"):
            with self.subTest(invalid=frequency), self.assertRaises(ValueError):
                filter_command(cast(int, frequency))

    def test_golden_single_all_32_channels(self):
        settings = MeasurementSettings(tuple(range(1, 33)), (1,) * 32, 10, 500)
        expected = (b":2" + b"00" * 32 + b"FFFFFFFF" + b"00000000"
                    + b"005A" + b"000000000000\n")
        self.assertEqual(configure_command(settings), expected)
        self.assertEqual(len(expected), 99)

    def test_golden_single_first_10_channels(self):
        settings = MeasurementSettings(tuple(range(1, 11)), (1,) * 32, 10, 250)
        expected = (b":2" + b"00" * 32 + b"000003FF" + b"00000000"
                    + b"0064" + b"000000000000\n")
        self.assertEqual(configure_command(settings), expected)
        self.assertEqual(len(expected), 99)

    def test_golden_sparse_physical_channels_1_3_32(self):
        delays = (1, 0, 2) + (0,) * 28 + (256,)
        settings = MeasurementSettings((1, 3, 32), delays, 13, 50)
        expected = (b":2" + b"000001" + b"00" * 28 + b"FF"
                    + b"80000005" + b"00000000" + b"001E"
                    + b"000000000000\n")
        command = configure_command(settings)
        self.assertEqual(command, expected)
        self.assertEqual(len(command), 99)
        self.assertEqual(command[66:74], b"80000005")
        self.assertEqual(command[74:82], b"00000000")
        self.assertEqual(command[86:98], b"000000000000")
        self.assertEqual(command[98:], b"\n")

    def test_delay_minus_one_and_disabled_delays_are_encoded(self):
        delays = (1, 2, 16, 17, 255, 256, 0) + (0,) * 25
        settings = MeasurementSettings((1,), delays, 1, 500)
        expected = (b":2" + b"00010F10FEFF00" + b"00" * 25
                    + b"00000001" + b"00000000" + b"000A"
                    + b"000000000000\n")
        self.assertEqual(settings.minimum_cycle_seconds, 0)
        self.assertEqual(configure_command(settings), expected)
        self.assertEqual(len(expected), 99)

    def test_cycle_duration_truncates_before_pause_multiplication(self):
        for delay, cycle, pause in ((24, 0, b"001E"), (25, 1, b"0014"),
                                    (26, 1, b"0014"), (49, 1, b"0014"),
                                    (50, 2, b"000A"), (51, 2, b"000A")):
            with self.subTest(delay=delay):
                settings = MeasurementSettings((1,), (delay,) + (256,) * 31,
                                               3, 500)
                self.assertEqual(settings.minimum_cycle_seconds, cycle)
                command = configure_command(settings)
                self.assertEqual(command[82:86], pause)
                self.assertEqual(len(command), 99)

    def test_minimum_pause_and_largest_valid_interval(self):
        for delay, interval, pause in ((25, 1, b"0001"), (49, 1, b"0001"),
                                       (1, 6553, b"FFFA")):
            with self.subTest(delay=delay, interval=interval):
                settings = MeasurementSettings((1,), (delay,) + (0,) * 31,
                                               interval, 500)
                self.assertEqual(configure_command(settings)[82:86], pause)
                self.assertEqual(len(configure_command(settings)), 99)

    def test_configure_rejects_unsupported_or_unencodable_settings(self):
        good: dict[str, Any] = {
            "channels": (1,), "channel_delays": (1,) * 32,
            "interval_seconds": 10, "filter_hz": 500,
        }
        bad_settings: tuple[dict[str, Any], ...] = (
            {"mode": "cross"}, {"channels": ()}, {"channels": (1, 1)},
            {"channels": (3, 1)}, {"channels": (0,)}, {"channels": (33,)},
            {"channels": (True,)}, {"channel_delays": (1,) * 31},
            {"channel_delays": (0,) * 32},
            {"channel_delays": (-1,) + (1,) * 31},
            {"channel_delays": (257,) + (1,) * 31},
            {"channel_delays": (True,) + (1,) * 31},
            {"interval_seconds": 6554},
        )
        for changes in bad_settings:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                configure_command(MeasurementSettings(**(good | changes)))


class SignedWordTests(unittest.TestCase):
    def test_signed_limits_and_lowercase(self):
        for field, expected in ((b"0000", 0), (b"0001", 1), (b"7FFF", 32767),
                                (b"8000", -32768), (b"8001", -32767),
                                (b"FFFE", -2), (b"FFFF", -1),
                                (b"fFfF", -1), (b"abcd", -21555)):
            with self.subTest(field=field):
                self.assertEqual(signed_word(field), expected)

    def test_invalid_word_length_and_non_hex(self):
        for field in (b"", b"0", b"123", b"12345", b"GGGG", b"0x10",
                      b"+001", b"-001", b" 001", b"001\n", b"\xff000"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                signed_word(field)


class DocumentFrameTests(unittest.TestCase):
    def assert_frame(self, frame, packet, first_ns, completed_ns):
        self.assertEqual(frame.reference_id, int(packet[2:4], 16))
        self.assertEqual(frame.values, expected_values(packet))
        self.assertIsInstance(frame.values, tuple)
        self.assertEqual(frame.raw, packet)
        self.assertEqual(frame.terminator, b"10")
        self.assertEqual(frame.first_received_ns, first_ns)
        self.assertEqual(frame.completed_ns, completed_ns)

    def test_fixture_integrity_and_all_102_real_packets(self):
        cases = (("cross32.txt", 32, list(range(32)),
                  "ef3cdf17605d097bb27095189318ac599023d65c889bf209200f51f3cb71d480"),
                 ("cross10.txt", 10, list(range(10)) * 7,
                  "caddf4a6b665d89aaec350aad033fd5049c394cfc2e9ddfb6a25ac27a55cdb67"))
        for name, count, references, digest in cases:
            with self.subTest(fixture=name):
                self.assertEqual(hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest(),
                                 digest)
                sample = packets(name)
                self.assertEqual(len(sample), len(references))
                parser = FrameParser(count, TERMINATORS)
                frames = parser.feed(b"".join(sample), received_ns=1234)
                self.assertEqual([f.reference_id for f in frames], references)
                for index, (frame, packet) in enumerate(zip(frames, sample)):
                    with self.subTest(fixture=name, packet=index):
                        self.assertEqual(len(packet), 6 + 4 * count)
                        self.assertEqual(packet[-2:], b"10")
                        self.assert_frame(frame, packet, 1234, 1234)
                self.assertEqual(parser.buffered_bytes, 0)
                self.assertEqual(parser.invalid_frames, 0)
                self.assertEqual(parser.discarded_bytes, 0)

    def test_first_document_packets_have_independent_golden_values(self):
        cases = (("cross32.txt", 32,
                  (4990, 2275, 2391, 2142, 1992, 2159, 2144, 2169,
                   2166, 2097, 2109, 1941, 2179, 1650, 2075, 1162,
                   1945, 1132, 1722, 973, 1899, 400, 1416, -197,
                   1005, -487, 39, -930, -220, -1653, -1961, -1835)),
                 ("cross10.txt", 10,
                  (4990, -1311, -1255, -1675, 84, -1775, -1577, -1769,
                   -1237, -2363)))
        for name, count, values in cases:
            with self.subTest(fixture=name):
                frame, = FrameParser(count, TERMINATORS).feed(packets(name)[0], 10)
                self.assertEqual(frame.reference_id, 0)
                self.assertEqual(frame.values, values)

    def test_every_split_boundary_of_first_cross10_packet(self):
        packet = packets("cross10.txt")[0]
        for split in range(len(packet) + 1):
            with self.subTest(split=split):
                parser = FrameParser(10, TERMINATORS)
                first = parser.feed(packet[:split], received_ns=100)
                second = parser.feed(packet[split:], received_ns=200)
                self.assertEqual(len(first), int(split == len(packet)))
                self.assertEqual(len(first + second), 1)
                self.assert_frame((first + second)[0], packet,
                                  200 if split == 0 else 100,
                                  100 if split == len(packet) else 200)
                self.assertEqual(parser.buffered_bytes, 0)

    def test_first_cross32_packet_byte_by_byte(self):
        packet = packets("cross32.txt")[0]
        parser = FrameParser(32, TERMINATORS)
        for index, byte in enumerate(packet):
            frames = parser.feed(bytes([byte]), received_ns=1000 + index)
            if index < len(packet) - 1:
                self.assertEqual(frames, [])
            else:
                self.assertEqual(len(frames), 1)
                self.assert_frame(frames[0], packet, 1000, 1000 + index)
        self.assertEqual(parser.buffered_bytes, 0)
        self.assertEqual(parser.invalid_frames, 0)

    def test_all_document_packets_in_repeated_small_chunks(self):
        for name, count in (("cross32.txt", 32), ("cross10.txt", 10)):
            stream = b"".join(packets(name))
            for chunk_size in (1, 2, 3, 7, 43, 134, 257):
                with self.subTest(fixture=name, chunk_size=chunk_size):
                    parser = FrameParser(count, TERMINATORS)
                    frames = []
                    for start in range(0, len(stream), chunk_size):
                        frames.extend(parser.feed(stream[start:start + chunk_size], start))
                    self.assertEqual([f.raw for f in frames], packets(name))
                    offset = 0
                    for frame in frames:
                        self.assertEqual(frame.values, expected_values(frame.raw))
                        self.assertEqual(frame.first_received_ns,
                                         offset // chunk_size * chunk_size)
                        offset += len(frame.raw)
                        self.assertEqual(frame.completed_ns,
                                         (offset - 1) // chunk_size * chunk_size)
                    self.assertEqual(parser.buffered_bytes, 0)
                    self.assertEqual(parser.invalid_frames, 0)
                    self.assertEqual(parser.discarded_bytes, 0)

    def test_multiple_frames_preserve_unfinished_next_frame(self):
        first, second, third = packets("cross10.txt")[:3]
        parser = FrameParser(10, TERMINATORS)
        frames = parser.feed(first + second + third[:17], 100)
        self.assertEqual(len(frames), 2)
        self.assert_frame(frames[0], first, 100, 100)
        self.assert_frame(frames[1], second, 100, 100)
        self.assertEqual(parser.buffered_bytes, 17)
        self.assertEqual(parser.feed(b"", 150), [])
        self.assertEqual(parser.buffered_bytes, 17)
        last, = parser.feed(third[17:], 200)
        self.assert_frame(last, third, 100, 200)
        self.assertEqual(parser.buffered_bytes, 0)

    def test_first_timestamp_allows_excluding_pretrigger_partial_frame(self):
        old, fresh = packets("cross10.txt")[:2]
        parser = FrameParser(10, TERMINATORS)
        self.assertEqual(parser.feed(old[:1], 90), [])
        trigger_ns = 100
        frames = parser.feed(old[1:] + fresh, 110)
        self.assertEqual([f.completed_ns for f in frames], [110, 110])
        self.assertEqual([f.first_received_ns for f in frames], [90, 110])
        eligible = [f for f in frames if f.first_received_ns >= trigger_ns]
        self.assertEqual([f.raw for f in eligible], [fresh])
        self.assertEqual([f.raw for f in frames], [old, fresh])


class ParserEdgeTests(unittest.TestCase):
    def test_all_terminators_and_partial_two_byte_terminators(self):
        body = packets("cross10.txt")[0][:-2]
        for term in TERMINATORS:
            for split in range(len(term) + 1):
                with self.subTest(term=term, split=split):
                    parser = FrameParser(10, TERMINATORS)
                    first = parser.feed(body + term[:split], 100)
                    second = parser.feed(term[split:], 200)
                    self.assertEqual(len(first), int(split == len(term)))
                    frame, = first + second
                    self.assertEqual(frame.raw, body + term)
                    self.assertEqual(frame.values, expected_values(body + b"10"))
                    self.assertEqual(frame.terminator, term)
                    self.assertEqual(frame.first_received_ns, 100)
                    self.assertEqual(frame.completed_ns,
                                     100 if split == len(term) else 200)
                    self.assertEqual(parser.buffered_bytes, 0)

    def test_mixed_terminators_in_one_read(self):
        sample = packets("cross10.txt")[:3]
        stream = b"".join(p[:-2] + term for p, term in zip(sample, TERMINATORS))
        frames = FrameParser(10, TERMINATORS).feed(stream, 123)
        self.assertEqual(len(frames), 3)
        self.assertEqual([f.terminator for f in frames], list(TERMINATORS))
        self.assertEqual(b"".join(f.raw for f in frames), stream)

    def test_signed_limits_and_reference_31_in_parser(self):
        packet = b":11F00007FFF8000FFFF8001fffe\n"
        frame, = FrameParser(6, TERMINATORS).feed(packet, 0)
        self.assertEqual(frame.reference_id, 31)
        self.assertEqual(frame.values, (0, 32767, -32768, -1, -32767, -2))
        self.assertEqual(frame.raw, packet)
        self.assertEqual(frame.first_received_ns, 0)
        self.assertEqual(frame.completed_ns, 0)

    def test_default_timestamps_use_host_monotonic_clock(self):
        packet = packets("cross10.txt")[0]
        parser = FrameParser(10, TERMINATORS)
        with patch("com_reader.protocol.time.monotonic_ns", side_effect=(10, 20)):
            self.assertEqual(parser.feed(packet[:5]), [])
            frame, = parser.feed(packet[5:])
        self.assertEqual(frame.first_received_ns, 10)
        self.assertEqual(frame.completed_ns, 20)

    def test_noise_and_split_header_resynchronize_with_correct_timestamp(self):
        packet = packets("cross10.txt")[0]
        noise = b"\x00\xffgarbage\r\n:AX\n:0:"
        parser = FrameParser(10, TERMINATORS)
        self.assertEqual(parser.feed(noise, 10), [])
        self.assertEqual(parser.buffered_bytes, 1)
        self.assertEqual(parser.discarded_bytes, len(noise) - 1)
        frame, = parser.feed(packet[1:], 20)
        self.assertEqual(frame.raw, packet)
        self.assertEqual(frame.first_received_ns, 10)
        self.assertEqual(frame.completed_ns, 20)
        self.assertEqual(parser.invalid_frames, 0)
        self.assertEqual(parser.buffered_bytes, 0)

    def test_invalid_hex_reference_and_terminator_resynchronize(self):
        packet = packets("cross10.txt")[0]
        corrupt = (
            packet[:4] + b"G" + packet[5:],
            packet[:4] + b"\xff" + packet[5:],
            packet[:2] + b"20" + packet[4:],
            packet[:2] + b"FF" + packet[4:],
            packet[:2] + b"-1" + packet[4:],
            packet[:-2] + b"XX",
        )
        for bad in corrupt:
            with self.subTest(packet=bad):
                parser = FrameParser(10, TERMINATORS)
                frames = parser.feed(bad + packet, 123)
                self.assertEqual([f.raw for f in frames], [packet])
                self.assertEqual(frames[0].values, expected_values(packet))
                self.assertGreaterEqual(parser.invalid_frames, 1)
                self.assertEqual(parser.discarded_bytes, len(bad))
                self.assertEqual(parser.buffered_bytes, 0)

    def test_invalid_frames_in_chunks_do_not_poison_next_timestamp(self):
        packet = packets("cross10.txt")[0]
        bad = packet[:2] + b"20" + packet[4:]
        parser = FrameParser(10, TERMINATORS)
        self.assertEqual(parser.feed(bad[:20], 10), [])
        self.assertEqual(parser.feed(bad[20:] + b"noise:", 20), [])
        frame, = parser.feed(packet[1:], 30)
        self.assertEqual(frame.raw, packet)
        self.assertEqual(frame.first_received_ns, 20)
        self.assertEqual(frame.completed_ns, 30)
        self.assertGreaterEqual(parser.invalid_frames, 1)

    def test_truncated_packet_resynchronizes_at_every_new_header_boundary(self):
        packet = packets("cross10.txt")[0]
        for cut in range(2, len(packet)):
            for same_read in (False, True):
                with self.subTest(cut=cut, same_read=same_read):
                    parser = FrameParser(10, TERMINATORS)
                    if same_read:
                        frames = parser.feed(packet[:cut] + packet, 20)
                    else:
                        self.assertEqual(parser.feed(packet[:cut], 10), [])
                        frames = parser.feed(packet, 20)
                    self.assertEqual([f.raw for f in frames], [packet])
                    self.assertEqual(frames[0].first_received_ns, 20)
                    self.assertEqual(frames[0].completed_ns, 20)
                    self.assertGreaterEqual(parser.invalid_frames, 1)
                    self.assertEqual(parser.discarded_bytes, cut)
                    self.assertEqual(parser.buffered_bytes, 0)

    def test_incomplete_packet_is_never_emitted(self):
        packet = packets("cross10.txt")[0]
        for cut in range(1, len(packet)):
            with self.subTest(cut=cut):
                parser = FrameParser(10, TERMINATORS)
                self.assertEqual(parser.feed(packet[:cut], 10), [])
                self.assertEqual(parser.feed(b"", 20), [])
                self.assertEqual(parser.buffered_bytes, cut)

    def test_constructor_validates_channel_count_and_nonempty_terminators(self):
        for count in (0, 33, -1, True, False, 10.0, "10", None):
            with self.subTest(count=count), self.assertRaises(ValueError):
                FrameParser(cast(int, count), TERMINATORS)
        for terms in ((), (b"",), (b"10", b"")):
            with self.subTest(terms=terms), self.assertRaises(ValueError):
                FrameParser(10, terms)
        for count in (1, 32):
            self.assertEqual(FrameParser(count, TERMINATORS).channel_count, count)


if __name__ == "__main__":
    unittest.main()
