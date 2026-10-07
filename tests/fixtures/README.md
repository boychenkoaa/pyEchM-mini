# Document packet fixtures

Source: `../../../данные.docx` (relative to this directory), read without modification.
Source SHA256: `f3f168900e5d230ecdb37907d7c8eea902df35f022fbc79e8ad9e7e0f9dc2fee`.

- `cross32.txt`: the `Перекрестный режим 32X32` sample, 32 packets,
  reference IDs 0..31, 32 signed 16-bit values per packet.
- `cross10.txt`: the `Перекрестный 10x10` sample, 70 packets,
  seven cycles of reference IDs 0..9, 10 signed 16-bit values per packet.

Extraction used only Python stdlib `zipfile`, `xml.etree.ElementTree` and `re`:
read `word/document.xml`, concatenate the text of all `w:t` elements in document
order, split at the two headings, and extract `:1`, two reference hex digits,
4 hex digits per channel, and the literal ASCII tail `10`. Joining the Word text
runs restores packets split by formatting (including reference 05 in Cross10).
Every byte in the two sample sections was accounted for; the trailing `:C` after
the Cross10 sample is not a measurement packet and was omitted.

Each ASCII file contains one unchanged complete packet per line. The added LF
is a fixture record separator, NOT the device terminator. Tests remove those
separators and concatenate packets as needed. Every original packet ends with
`b'10'` (ASCII bytes `31 30`), not byte `0x10` or LF. Tests require neither the
DOCX nor any external package, device, serial port, or network connection.

SHA256 of fixture files (ASCII with LF separators and a final LF):

- `cross32.txt`: `ef3cdf17605d097bb27095189318ac599023d65c889bf209200f51f3cb71d480`
- `cross10.txt`: `caddf4a6b665d89aaec350aad033fd5049c394cfc2e9ddfb6a25ac27a55cdb67`
