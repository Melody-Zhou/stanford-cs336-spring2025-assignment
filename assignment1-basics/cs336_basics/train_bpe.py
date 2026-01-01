import os
from collections import Counter
from multiprocessing import Pool
from cs336_basics.pretokenization_example import find_chunk_boundaries

# GPT-2-style regex pre-tokenizer (requires third-party `regex`)
GPT2_PRETOKENIZE_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# Each worker process compiles to its own regex instance
RX = None

def init_worker():
    global RX
    import regex as re
    RX = re.compile(GPT2_PRETOKENIZE_PATTERN)

def count_word_freq_from_text(text: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    """
    Perform pre-tokenization on a text chunk:
      1) Split on special tokens (special tokens do not participate in training stats)
      2) Apply GPT-2 regex pre-tokenization
      3) Encode each piece to UTF-8 bytes and split into single-byte tokens
      4) Count token-sequence frequencies
    """
    if not text:
        return {}

    # split by each special token; drop the special tokens from training stats
    spans = [text]
    for s_tok in special_tokens:
        new_spans: list[str] = []
        for sp in spans:
            if sp:
                new_spans.extend(sp.split(s_tok))
        spans = new_spans
    
    word_freq: dict[tuple[bytes, ...], int] = {}
    for sp in spans:
        if not sp:
            continue
        for m in RX.finditer(sp):
            piece = m.group(0)
            if not piece:
                continue
            bts = piece.encode("utf-8")
            key = tuple(bytes([b]) for b in bts)
            word_freq[key] = word_freq.get(key, 0) + 1
    return word_freq

def process_chunk(args) -> dict[tuple[bytes, ...], int]:
    """
    Worker entry point for processing a single file chunk.
    """
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start)

    # Decode with errors ignored to avoid UTF-8 boundary issues
    text = chunk.decode("utf-8", errors="ignore")
    return count_word_freq_from_text(text, special_tokens)

def build_word_freq_serial(input_path : str | os.PathLike, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    init_worker()
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    return count_word_freq_from_text(text, special_tokens)

def build_word_freq_parallel(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_processes: int,
    *,
    num_chunks: int | None = None
) -> dict[tuple[bytes, ...], int]:
    """
    Build word frequency statistics using multiprocessing.
    Chunk boundaries are aligned to special-token boundaries.    
    """
    if num_processes <= 1 or not special_tokens:
        return build_word_freq_serial(input_path, special_tokens)
    
    if num_chunks is None:
        num_chunks = max(num_processes * 32, num_processes)

    split_special_token = special_tokens[0].encode("utf-8")  # e.g. b"<|endoftext|>"
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, split_special_token)
    
    tasks = [(str(input_path), s, e, special_tokens) for s, e in zip(boundaries[:-1], boundaries[1:])]

    merged = Counter()
    with Pool(processes=num_processes, initializer=init_worker, maxtasksperchild=8) as pool:
        for partial in pool.imap_unordered(process_chunk, tasks, chunksize=1):
            merged.update(partial)
    
    return dict(merged)

def pairs_in_word(word: tuple[bytes, ...]) -> dict[tuple[bytes, bytes], int]:
    """
    Count adjacent pair occurrences within a single word sequence.
    """
    counts: dict[tuple[bytes, bytes], int] = {}
    if len(word) < 2:
        return counts
    prev = word[0]
    for cur in word[1:]:
        p = (prev, cur)
        counts[p] = counts.get(p, 0) + 1
        prev = cur
    return counts

def apply_merge(word: tuple[bytes, ...], a: bytes, b: bytes, new_token: bytes) -> tuple[bytes, ...]:
    """
    Replace occurrences of (a,b) with new_token.
    """
    if len(word) < 2:
        return word
    merged: list[bytes] = []
    i = 0
    L = len(word)
    while i < L:
        if i < L - 1 and word[i] == a and word[i + 1] ==b:
            merged.append(new_token)
            i += 2
        else:
            merged.append(word[i])
            i += 1
    return tuple(merged)

def build_pair_stats(
    word_freq: dict[tuple[bytes, ...], int]
) -> tuple[dict[tuple[bytes, bytes], int], dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]]:
    """
    Build:
      - pair_counts: global weighted counts for each adjacent pair
      - pair_to_words: inverted index (pair -> set of words containing that pair)
    """
    pair_counts: dict[tuple[bytes, bytes], int] = {}
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = {}

    for word, freq in word_freq.items():
        if len(word) < 2:
            continue
        local = pairs_in_word(word)
        for p, occ in local.items():
            pair_counts[p] = pair_counts.get(p, 0) + occ * freq
            s = pair_to_words.get(p)
            if s is None:
                pair_to_words[p] = {word}
            else:
                s.add(word)
    
    return pair_counts, pair_to_words

def remove_word_contrib(
    word: tuple[bytes, ...],
    freq: int,
    pair_counts: dict[tuple[bytes, bytes], int],
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
) -> None:
    """
    Remove a word's contribution from pair_counts and pair_to_words.
    """
    local = pairs_in_word(word)
    for p, occ in local.items():
        s = pair_to_words.get(p)
        if s is not None:
            s.discard(word)
            if not s:
                del pair_to_words[p]
        
        new_c = pair_counts.get(p, 0) - occ * freq
        if new_c <= 0:
            pair_counts.pop(p, None)
        else:
            pair_counts[p] = new_c

def add_word_contrib(
    word: tuple[bytes, ...],
    add_freq: int,
    pair_counts: dict[tuple[bytes, bytes], int],
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
    *,
    word_is_new: bool,
) -> None:
    """
    Add a word's contribution to pair_counts and pair_to_words.
    """
    if len(word) < 2:
        return
    local = pairs_in_word(word)
    for p, occ in local.items():
        pair_counts[p] = pair_counts.get(p, 0) + occ * add_freq
        if word_is_new:
            s = pair_to_words.get(p)
            if s is None:
                pair_to_words[p] = {word}
            else:
                s.add(word)

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    *,
    num_processes: int | None = None
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer.

    Returns:
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]    
    """
    if vocab_size <= 0:
        raise ValueError("vocab_size must be a positive integer")
    if vocab_size < 256 + len(special_tokens):
        raise ValueError("vocab_size too small: must be >= 256 + len(special_tokens)")

    # ---- vocab init: 256 single-byte tokens + special tokens ----
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1
    
    # ---- pre-tokenization + counting (parallelization) ----
    if num_processes is None:
        num_processes = min(8, os.cpu_count() or 1)
    
    file_size = os.path.getsize(input_path)

    # # For small files, multiprocessing overhead dominates; use serial
    if num_processes <= 1 or file_size < 1_000_000:  # ~1MB
        word_freq = build_word_freq_serial(input_path, special_tokens)
    else:
        word_freq = build_word_freq_parallel(input_path, special_tokens, num_processes, num_chunks=num_processes * 32)
    
    if not word_freq:
        return vocab, []

    # ---- BPE merges ----
    pair_counts, pair_to_words = build_pair_stats(word_freq)
    merges: list[tuple[bytes, bytes]] = []
    while next_id < vocab_size:
        if not pair_counts:
            break
    
        # choose most frequent; tie-break by lexicographically largest pair
        (a, b), best_count = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))
        if best_count <= 0:
            break
        
        new_token = a + b
        merges.append((a, b))
        vocab[next_id] = new_token
        next_id += 1

        affected = pair_to_words.get((a, b))
        if not affected:
            pair_counts.pop((a, b), None)
            continue

        # replace occurrences of (a,b) in every word
        add_back: dict[tuple[bytes, ...], int] = {}
        for word in list(affected):
            freq = word_freq.get(word)
            if freq is None:
                continue

            remove_word_contrib(word, freq, pair_counts, pair_to_words)
            del word_freq[word]

            new_word = apply_merge(word, a, b, new_token)
            add_back[new_word] = add_back.get(new_word, 0) + freq
        
        for new_word, add_freq in add_back.items():
            existed = new_word in word_freq
            word_freq[new_word] = word_freq.get(new_word, 0) + add_freq
            add_word_contrib(new_word, add_freq, pair_counts, pair_to_words, word_is_new=not existed)

    return vocab, merges