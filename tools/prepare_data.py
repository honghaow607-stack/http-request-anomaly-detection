"""Prepare the fixed reference corpus or the bundled synthetic demonstration.

The manifest records the original input bytes. This wrapper does not change the
request parser, feature pipeline, split, estimators, or reported metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_file(path: Path, expected: str, lf_expected: str | None = None) -> None:
    if sha256(path) == expected:
        return
    # The parser uses universal newlines. Verified mirrors may store LF while
    # the original Windows files contain CRLF or a mixture of newline styles.
    if lf_expected is not None:
        normalized = path.read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(normalized).hexdigest() == lf_expected:
            return
    raise ValueError(f"SHA256 mismatch: {path.name}. Use the manifest's original file.")


def download_file(record: dict, destination: Path) -> None:
    errors: list[str] = []
    for url in record["mirrors"]:
        temporary: Path | None = None
        try:
            request = Request(url, headers={"User-Agent": "HTTP-Request-Anomaly-Detection"})
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
                temporary = Path(handle.name)
                with urlopen(request, timeout=25) as response:
                    shutil.copyfileobj(response, handle)
            check_file(temporary, record["sha256"], record.get("lf_sha256"))
            temporary.replace(destination)
            return
        except (OSError, URLError, ValueError) as error:
            errors.append(f"{url}: {type(error).__name__}: {error}")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    raise RuntimeError(
        f"Could not obtain the verified file {record['name']}. "
        "Use --source-dir with the original data/raw folder. "
        "No synthetic fallback was substituted.\n" + "\n".join(errors)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["paper", "synthetic"], default="paper")
    parser.add_argument("--source-dir", type=Path, help="Folder containing the two original CSIC files")
    args = parser.parse_args()

    manifest = json.loads((PROJECT_ROOT / "docs/data_manifest.json").read_text(encoding="utf-8"))
    selected = [item for item in manifest["files"] if args.mode == "paper" or item["kind"] == "synthetic"]
    names = {item["name"] for item in selected}
    raw_dir = PROJECT_ROOT / "data/raw"
    extras = sorted(path.name for path in raw_dir.glob("*.txt") if path.name not in names)
    if extras:
        raise ValueError(
            "Extra raw files would change the experiment: " + ", ".join(extras)
            + ". Back up and move them out of data/raw before switching modes."
        )

    # Validate existing inputs before writing or downloading anything.
    for item in selected:
        destination = raw_dir / item["name"]
        if destination.exists():
            check_file(destination, item["sha256"], item.get("lf_sha256"))
        elif item["kind"] == "synthetic":
            check_file(PROJECT_ROOT / item["bundled_path"], item["sha256"])
        elif args.source_dir is not None:
            check_file(args.source_dir / item["name"], item["sha256"], item.get("lf_sha256"))

    raw_dir.mkdir(parents=True, exist_ok=True)
    for item in selected:
        destination = raw_dir / item["name"]
        if not destination.exists():
            if item["kind"] == "synthetic":
                shutil.copyfile(PROJECT_ROOT / item["bundled_path"], destination)
            elif args.source_dir is not None:
                shutil.copyfile(args.source_dir / item["name"], destination)
            else:
                print(f"Downloading {item['name']} ...", flush=True)
                download_file(item, destination)
        print(f"Verified {item['name']}")

    from src.data_loader import build_dataset, ensure_dirs

    ensure_dirs()
    dataset = build_dataset(save=True)
    expected_rows = sum(item["retained_requests"] for item in selected)
    if len(dataset) != expected_rows:
        raise ValueError(f"Unexpected parsed size: {len(dataset)}; expected {expected_rows}.")
    print(f"\nPrepared mode={args.mode}: {len(dataset):,} retained requests.")
    print("Next: python run.py")
    if args.mode == "synthetic":
        print("Synthetic demonstration only; these are not the reference experiment's inputs.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Data preparation failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
