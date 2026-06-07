import argparse
import math
from pathlib import Path
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "proxectonos/Carballo-bloom-1.3B"


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    return tokenizer, model


def compute_perplexity(tokenizer, model, text: str) -> float:
    """Compute perplexity for a single line."""
    if not text.strip():
        return float("nan")

    encodings = tokenizer(text, return_tensors="pt")

    input_ids = encodings.input_ids
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    return math.exp(loss.item())


def process_file(tokenizer, model, input_path: Path, output_path: Path) -> float:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    perplexities: List[float] = []
    rows = []

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            ppl = compute_perplexity(tokenizer, model, line)
            if not math.isnan(ppl):
                perplexities.append(ppl)
            rows.append((line, ppl))

    avg_ppl = sum(perplexities) / len(perplexities) if perplexities else float("nan")

    with output_path.open("w", encoding="utf-8") as f:
        f.write("translation\tperplexity\n")
        for line, ppl in rows:
            f.write(f"{line}\t{ppl:.6f}\n")

        #f.write(f"AVERAGE\t{avg_ppl:.6f}\n")

    return avg_ppl


def process_all(best_dir: Path, output_root_name: str, suffix: str) -> None:
    tokenizer, model = load_model()

    detok_files = sorted(best_dir.rglob("*.pt.detok"))
    if not detok_files:
        print(f"No .pt.detok files found under {best_dir}")
        return

    global_rows = []

    for detok_file in detok_files:
        run_root = next((p for p in detok_file.parents if p.parent == best_dir), None)
        if run_root is None:
            print(f"Skipping {detok_file}: cannot infer run root")
            continue

        rel_to_run = detok_file.relative_to(run_root)
        out_file = run_root / output_root_name / rel_to_run.with_suffix(f"{suffix}.tsv")
        avg_ppl = process_file(tokenizer, model, detok_file, out_file)

        row = {
            "run": run_root.name,
            "file": str(rel_to_run),
            "avg_perplexity": f"{avg_ppl:.6f}",
            "ppl_file": str(out_file.relative_to(best_dir)),
        }
        global_rows.append(row)
        print(f"{run_root.name}\t{rel_to_run}\t{avg_ppl:.6f}")

    if global_rows:
        global_summary = best_dir / "ppl_summary.tsv"
        with global_summary.open("w", encoding="utf-8") as f:
            f.write("run\tfile\tavg_perplexity\tppl_file\n")
            for row in global_rows:
                f.write(
                    f"{row['run']}\t{row['file']}\t{row['avg_perplexity']}\t{row['ppl_file']}\n"
                )
        print(f"Wrote {global_summary}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute perplexity for .pt.detok files.")
    parser.add_argument(
        "--best-dir",
        default="/home/compartido/daniel/2026/best",
        help="Directory containing run_* folders and .pt.detok files.",
    )
    parser.add_argument(
        "--output-root",
        default="ppl",
        help="Per-run output subfolder for line-level perplexity TSV files.",
    )
    parser.add_argument(
        "--suffix",
        default=".ppl",
        help="Suffix added before .tsv in generated output filenames.",
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        help="Optional single-file input path (legacy mode).",
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        help="Optional single-file output path (legacy mode).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.input_path or args.output_path:
        if not (args.input_path and args.output_path):
            raise SystemExit("For single-file mode, provide both input_path and output_path")
        tokenizer, model = load_model()
        process_file(tokenizer, model, Path(args.input_path), Path(args.output_path))
        print(f"Wrote {args.output_path}")
        return

    process_all(Path(args.best_dir), args.output_root, args.suffix)


if __name__ == "__main__":
    main()
