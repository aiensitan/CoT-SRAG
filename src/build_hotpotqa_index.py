import json
import os
import subprocess
from pathlib import Path


def build_raw_jsonl(src_path: Path, dst_path: Path) -> None:
    data = json.loads(src_path.read_text(encoding="utf-8"))
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open("w", encoding="utf-8") as f:
        for d in data:
            parts = []
            for item in d.get("context", []):
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    title, sents = item
                    if isinstance(sents, list):
                        parts.append(f"{title}: " + " ".join(sents))
                    else:
                        parts.append(f"{title}: {sents}")
                else:
                    parts.append(str(item))
            text = "\n".join(parts)
            f.write(json.dumps({"context": text}, ensure_ascii=False) + "\n")


def run_gen_index(project_root: Path) -> None:
    cmd = [
        "python",
        "gen_index.py",
        "--dataset",
        "hotpotqa",
        "--chunk_size",
        "200",
        "--min_sentence",
        "2",
        "--overlap",
        "2",
        "--out_root",
        "../data/corpus/processed",
    ]
    subprocess.check_call(cmd, cwd=str(project_root / "src"))


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "data" / "hotpotqa" / "hotpot_dev_distractor_v1_dev_1000.json"
    dst_path = project_root / "data" / "corpus" / "raw" / "hotpotqa.jsonl"

    if not src_path.exists():
        raise FileNotFoundError(f"Missing input file: {src_path}")

    print(f"[1/2] Building raw jsonl: {dst_path}")
    build_raw_jsonl(src_path, dst_path)

    print("[2/2] Building index via gen_index.py")
    run_gen_index(project_root)

    print("Done.")


if __name__ == "__main__":
    main()
