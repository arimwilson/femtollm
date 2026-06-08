"""A small byte-pair encoding (BPE) tokenizer.

This is intentionally compact and dependency-free for assignment work. It starts
from UTF-8 bytes, learns frequent adjacent token merges, then applies those
merges in rank order when encoding new text.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


TokenId = int
Pair = tuple[TokenId, TokenId]


def _count_pairs(ids: list[TokenId]) -> Counter[Pair]:
    """Count adjacent token pairs in a token sequence."""
    return Counter(zip(ids, ids[1:]))


def _merge_pair(ids: list[TokenId], pair: Pair, new_id: TokenId) -> list[TokenId]:
    """Replace every non-overlapping occurrence of pair with new_id."""
    merged: list[TokenId] = []
    i = 0

    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            merged.append(new_id)
            i += 2
        else:
            merged.append(ids[i])
            i += 1

    return merged


@dataclass
class BPEEncoder:
    """Simple byte-level BPE encoder.

    Example:
        >>> enc = BPEEncoder(vocab_size=270)
        >>> enc.train("hello hello")
        >>> ids = enc.encode("hello")
        >>> enc.decode(ids)
        'hello'
    """

    vocab_size: int = 300
    vocab: dict[TokenId, bytes] = field(default_factory=dict)
    merges: dict[Pair, TokenId] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 for byte-level BPE")

        if not self.vocab:
            self.vocab = {i: bytes([i]) for i in range(256)}

    def train(self, text: str) -> None:
        """Learn BPE merges from text.

        Training resets any existing learned merges while preserving the base
        byte vocabulary. Merging stops early if no pair occurs more than once.
        """
        ids = list(text.encode("utf-8"))
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}

        while len(self.vocab) < self.vocab_size:
            pair_counts = _count_pairs(ids)
            if not pair_counts:
                break

            pair, count = pair_counts.most_common(1)[0]
            if count < 2:
                break

            new_id = len(self.vocab)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            ids = _merge_pair(ids, pair, new_id)

    def encode(self, text: str) -> list[TokenId]:
        """Encode text into BPE token ids."""
        ids = list(text.encode("utf-8"))

        while len(ids) >= 2:
            pairs = _count_pairs(ids)
            ranked_pairs = {
                pair: self.merges[pair] for pair in pairs if pair in self.merges
            }
            if not ranked_pairs:
                break

            # Lower token ids were learned earlier, so they have higher priority.
            pair = min(ranked_pairs, key=ranked_pairs.get)
            ids = _merge_pair(ids, pair, self.merges[pair])

        return ids

    def decode(self, ids: list[TokenId]) -> str:
        """Decode BPE token ids back into text."""
        try:
            raw = b"".join(self.vocab[token_id] for token_id in ids)
        except KeyError as exc:
            raise ValueError(f"unknown token id: {exc.args[0]}") from exc

        return raw.decode("utf-8", errors="replace")


if __name__ == "__main__":
    print("Understanding unicode/null chars part of the assignment")
    a = 'this is a test' + chr(0) + ' string'
    print('this is a test' + chr(0) + ' string')

    print("Understanding utf-8 part of the assignment")
    b = 'this is a test with an emoji: 😀'
    utf8_b = b.encode('utf-8')
    print(utf8_b)
    print(len(utf8_b))
    print(utf8_b.decode('utf-8')) 
    print(len(b))
    def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
        return "".join([bytes([b]).decode("utf-8") for b in bytestring])
    # This doesn't work because you can't decode each byte of a utf8 byte stream
    # indepdently; you have to decode them together (utf8 uses up to 4 bytes)
    #print(decode_utf8_bytes_to_str_wrong(utf8_b))
    utf8_undecodeable_bytes = bytes([0x8f, 0xff])
    print(utf8_undecodeable_bytes.decode('utf-8'))
    """sample = "the quick brown fox jumps over the quick brown dog"
    encoder = BPEEncoder(vocab_size=275)
    encoder.train(sample)

    encoded = encoder.encode("the quick brown")
    print("encoded:", encoded)
    print("decoded:", encoder.decode(encoded))"""
