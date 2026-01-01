import os
import pickle
import regex as re
from typing import Any, Iterable, Iterator

GPT2_PRETOKENIZE_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    """
    Byte-level BPE tokenizer compatible with GPT-2-style pre-tokenization.
    Vocab maps token_id -> bytes. Merges are list[(bytes, bytes)] in creation order.
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab: dict[int, bytes] = vocab
        self.merges: list[tuple[bytes, bytes]] = merges
        
        # reverse vocab: bytes -> id
        self.byte_to_id: dict[bytes, int] = {b: i for i, b in self.vocab.items()}

        # merge ranks: lower rank = higher priority
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {
            pair: idx for idx, pair in enumerate(self.merges)
        }

        # regex for GPT-2 pre-tokenization
        self.rx = re.compile(GPT2_PRETOKENIZE_PATTERN)

        # special tokens
        self.special_tokens: list[str] = special_tokens or []
        self.special_bytes: list[bytes] = []
        self.special_id: dict[str, int] = {}

        if self.special_tokens:
            # append missing special tokens to vocab
            for s in self.special_tokens:
                b = s.encode("utf-8")
                if b not in self.byte_to_id:
                    new_id = len(self.vocab)
                    self.vocab[new_id] = b
                    self.byte_to_id[b] = new_id
                self.special_id[s] = self.byte_to_id[b]
                self.special_bytes.append(b)
            
            # build a "longest-first" special-token matcher
            # we keep them as strings for boundary-safe matching in encode()
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            self._special_re = re.compile("|".join(re.escape(s) for s in sorted_special))
            self._max_special_len = max(len(s) for s in self.special_tokens)
        else:
            self._special_re = None
            self._max_special_len = 0

        # cache for BPE on pre-token byte sequence
        self._bpe_cache: dict[bytes, list[bytes]] = {}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None
    ) -> "Tokenizer":
        # match the training outputs used earlier (pickle dump)
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
    
    # ---------------------------
    # Public API
    # ---------------------------
    def encode(self, text: str) -> list[int]:
        if not text:
            return []    
        
        ids: list[int] = []
        if not self._special_re:
            ids.extend(self._encode_plain(text))
            return ids
        
        # split text by special tokens while preserving them as standalone parts
        last = 0
        for m in self._special_re.finditer(text):
            if m.start() > last:
                ids.extend(self._encode_plain(text[last : m.start()]))
            s = m.group(0)
            ids.append(self.special_id[s])
            last = m.end()
        if last < len(text):
            ids.extend(self._encode_plain(text[last:]))
        return ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Memory-efficient streaming encode that matches Tokenizer.encode(full_text)
        """
        buf = ""

        for chunk in iterable:
            if not chunk:
                continue
            buf += chunk

            while True:
                matches = list(self.rx.finditer(buf))
                if len(matches) <= 1:
                    break

                # keep the last match unprocessed; emit everything before it.
                cut = matches[-1].start()
                if cut <= 0:
                    break

                process_part = buf[:cut]
                buf = buf[cut:]

                for _id in self.encode(process_part):
                    yield _id
        
        # flush the remainder
        if buf:
            for _id in self.encode(buf):
                yield _id

    def decode(self, ids: list[int]) -> str:
        if not ids:
            return ""            
        b = b"".join(self.vocab[i] for i in ids)
        return b.decode("utf-8", errors="replace")

    # ---------------------------
    # Internal helpers
    # ---------------------------
    def _encode_plain(self, text: str) -> list[int]:
        """
        Encode a piece of text with no special tokens inside it.
        """
        out: list[int] = []
        for m in self.rx.finditer(text):
            piece = m.group(0)
            if not piece:
                continue
            piece_bytes = piece.encode("utf-8")
            for tok_bytes in self._bpe(piece_bytes):
                out.append(self.byte_to_id[tok_bytes])
        return out
    
    def _bpe(self, token_bytes: bytes) -> list[bytes]:
        """
        Apply BPE merges (by rank) on a single pre-token byte sequence.
        Returns a list of vocab byte tokens.        
        """
        cached = self._bpe_cache.get(token_bytes)
        if cached is not None:
            return cached
        
        # start from single bytes
        word: list[bytes] = [bytes([b]) for b in token_bytes]
        if len(word) <= 1:
            self._bpe_cache[token_bytes] = word
            return word
        
        while True:
            best_pair = None
            best_rank = None

            # find best ranked pair among adjacent pairs
            prev = word[0]
            for cur in word[1:]:
                p = (prev, cur)
                r = self.merge_ranks.get(p)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_pair = p
                prev = cur
            
            if best_pair is None:
                break
        
            a, b = best_pair
            new_token = a + b

            # merge all occurrences of (a, b)
            merged: list[bytes] = []
            i = 0
            L = len(word)
            while i < L:
                if i < L - 1 and word[i] == a and word[i + 1] == b:
                    merged.append(new_token)
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            word = merged
            if len(word) <= 1:
                break
        
        self._bpe_cache[token_bytes] = word
        return word