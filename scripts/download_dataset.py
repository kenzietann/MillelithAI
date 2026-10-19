from pathlib import Path

from datasets import load_dataset

OUTPUT_PATH = Path("data/raw/fineweb_edu_300mb.txt")
NUM_DOCUMENTS = 65000  # about 4.7 KB per document, so roughly 300 MB
PROGRESS_EVERY = 5000


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # streaming=True means we read documents one by one instead of downloading the whole dataset
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    documents_written = 0
    with open(OUTPUT_PATH, "w", encoding="utf-8") as output_file:
        for document in dataset:
            if documents_written >= NUM_DOCUMENTS:
                break
            output_file.write(document["text"] + "\n\n")
            documents_written += 1
            if documents_written % PROGRESS_EVERY == 0:
                print(f"{documents_written} / {NUM_DOCUMENTS} documents", flush=True)

    size_in_mb = OUTPUT_PATH.stat().st_size / 1_000_000
    if documents_written == 0 or size_in_mb == 0:
        raise RuntimeError("Nothing was written, check the dataset name and field")

    print(f"Wrote {documents_written} documents to {OUTPUT_PATH} ({size_in_mb:.1f} MB)")


if __name__ == "__main__":
    main()
