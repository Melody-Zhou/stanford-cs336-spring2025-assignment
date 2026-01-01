import os
import time
import pickle
import psutil

from cs336_basics.train_bpe import train_bpe

def main():
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    output_dir = "workspace"
    os.makedirs(output_dir, exist_ok=True)

    vocab_size = 10_000
    special_tokens = ["<|endoftext|>"]

    proc = psutil.Process(os.getpid())
    peak_rss = 0

    t0 = time.perf_counter()
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        num_processes=8
    )
    t1 = time.perf_counter()

    peak_rss = proc.memory_info().rss

    # Save to disk
    vocab_path = os.path.join(output_dir, "tinystories_bpe_vocab.pkl")
    merges_path = os.path.join(output_dir, "tinystories_bpe_merges.pkl")
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)

    # Longest token in vocab (by byte length)
    longest_id, longest_bytes = max(vocab.items(), key=lambda kv: len(kv[1]))
    longest_str = longest_bytes.decode("utf-8", errors="replace")

    elapsed_s = t1 - t0
    elapsed_min = elapsed_s / 60.0
    elapsed_hr  = elapsed_s / 3600.0
    mem_gb = peak_rss / (1024**3) 

    print(f"Saved vocab -> {vocab_path}")
    print(f"Saved merges -> {merges_path}")
    print(f"Elapsed: {elapsed_s:.2f}s ({elapsed_min:.2f} min, {elapsed_hr:.4f} hr)")
    print(f"RSS (approx): {mem_gb:.2f} GB  (install psutil for this number)")
    print(f"Longest token id={longest_id}, bytes_len={len(longest_bytes)}")
    print(f"Longest token (decoded): {repr(longest_str)}")

if __name__ == "__main__":
    main()