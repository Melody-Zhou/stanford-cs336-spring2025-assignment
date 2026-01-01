import time
import random
from typing import Iterator, TextIO

from cs336_basics.tokenizer import Tokenizer

EOT = "<|endoftext|>"

def iter_docs_by_eot(f: TextIO, eot: str) -> Iterator[str]:
    buf = ""
    for chunk in f:
        buf += chunk
        while True:
            idx = buf.find(eot)
            if idx < 0:
                break
            doc = buf[:idx]
            buf = buf[idx + len(eot) :]
            if doc.strip():
                yield doc
    # Emit the tail as the last doc (if non-empty).
    if buf.strip():
        yield buf


def reservoir_sample_docs(path: str, eot: str, k: int, seed: int) -> list[str]:
    rnd = random.Random(seed)
    sample: list[str] = []
    n = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for doc in iter_docs_by_eot(f, eot):
            n += 1
            if len(sample) < k:
                sample.append(doc)
            else:
                j = rnd.randrange(n)
                if j < k:
                    sample[j] = doc
    return sample

def bytes_per_token(tokenizer: Tokenizer, docs: list[str]) -> tuple[float, int, int]:
    total_bytes = 0
    total_tokens = 0
    for d in docs:
        b = len(d.encode("utf-8"))
        ids = tokenizer.encode(d)
        total_bytes += b
        total_tokens += len(ids)
    bpt = total_bytes / max(1, total_tokens)
    return bpt, total_bytes, total_tokens

def iter_first_n_bytes_as_lines(path: str, nbytes: int) -> Iterator[str]:
    seen = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line:
                break
            seen += len(line.encode("utf-8"))
            yield line
            if seen >= nbytes:
                break

def measure_throughput_bytes_per_sec(tokenizer: Tokenizer, it: Iterator[str], repeats: int = 1) -> float:
    lines = list(it)
    total_bytes = sum(len(x.encode("utf-8")) for x in lines)

    t0 = time.perf_counter()
    for _ in range(repeats):
        for _id in tokenizer.encode_iterable(lines):
            pass
    t1 = time.perf_counter()

    secs = (t1 - t0) / max(1, repeats)
    return total_bytes / max(1e-9, secs)

def main():
    # 1) paths to trained tokenizers
    tin_vocab_path = "workspace/tinystories_bpe_vocab.pkl"
    tin_merges_path = "workspace/tinystories_bpe_merges.pkl"

    owt_vocab_path = "workspace/owt_bpe_vocab_32000.pkl"
    owt_merges_path = "workspace/owt_bpe_merges_32000.pkl"

    # 2) paths to datasets
    tinystories_path = "data/TinyStoriesV2-GPT4-train.txt"
    owt_path = "data/owt_train.txt"

    # 3) load tokenizers
    tin_tok = Tokenizer.from_files(tin_vocab_path, tin_merges_path, special_tokens=[EOT])
    owt_tok = Tokenizer.from_files(owt_vocab_path, owt_merges_path, special_tokens=[EOT])

    # 4) smple 10 documents from each corpus WITHOUT loading the full file
    seed = 42
    print("Sampling TinyStories docs (streaming)...")
    tin_docs = reservoir_sample_docs(tinystories_path, EOT, k=10, seed=seed)
    print("Sampling OWT docs (streaming)...")
    owt_docs = reservoir_sample_docs(owt_path, EOT, k=10, seed=seed)

    # (a) In-domain compression efficiency
    tin_on_tin, tin_bytes, tin_tokens = bytes_per_token(tin_tok, tin_docs)
    owt_on_owt, owt_bytes, owt_tokens = bytes_per_token(owt_tok, owt_docs)

    # (b) cross-domain compression efficiency
    tin_on_owt, x_bytes, x_tokens = bytes_per_token(tin_tok, owt_docs)

    print("\n=== (a) In-domain bytes/token ===")
    print(f"TinyStories tokenizer on TinyStories: {tin_on_tin:.4f} bytes/token "
          f"(bytes={tin_bytes}, tokens={tin_tokens})")
    print(f"OWT tokenizer on OWT:               {owt_on_owt:.4f} bytes/token "
          f"(bytes={owt_bytes}, tokens={owt_tokens})")

    print("\n=== (b) Cross-domain bytes/token ===")
    print(f"TinyStories tokenizer on OWT:       {tin_on_owt:.4f} bytes/token "
          f"(bytes={x_bytes}, tokens={x_tokens})")

    # (c) Throughput estimation
    # use a small prefix of the corpus to avoid long measurement times

    sample_for_speed = iter_first_n_bytes_as_lines(owt_path, nbytes=5_000_000)  # 5MB
    thr = measure_throughput_bytes_per_sec(owt_tok, sample_for_speed, repeats=1)
    total_bytes_82gb = 82 * (1024 ** 3)
    est_secs = total_bytes_82gb / thr
    est_hours = est_secs / 3600

    print("\n=== (c) Throughput estimate ===")
    print(f"Measured throughput (OWT tokenizer): {thr:.2f} bytes/s")
    print(f"Estimated time for 82GB: {est_hours:.2f} hours")

if __name__ == "__main__":
    main()