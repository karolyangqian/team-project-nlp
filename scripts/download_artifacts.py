"""Download and extract pre-trained DRAFT model artifacts from Google Drive."""

import argparse
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactBundle:
    """Metadata describing one downloadable Google Drive artifact archive."""

    key: str
    gdrive_file_id: str
    size_mb: float
    target_subdir: str
    marker_relpath: str
    description: str


# Public Google Drive file IDs ("Anyone with the link").
ARTIFACT_BUNDLES: dict[str, ArtifactBundle] = {
    "classifier_baseline": ArtifactBundle(
        key="classifier_baseline",
        gdrive_file_id="1yaTsjZjV1Gszmu02EltGubgyNZzkj-61",
        size_mb=3.2,
        target_subdir="classifier_baseline",
        marker_relpath="vectorizer.joblib",
        description="Stage 1 TF-IDF + LogisticRegression baseline",
    ),
    "retriever": ArtifactBundle(
        key="retriever",
        gdrive_file_id="1NG8N1d4sFlQufDfqLPOhf6SgeaULsWkc",
        size_mb=139.0,
        target_subdir="retriever",
        marker_relpath="index/index_meta.json",
        description="Stage 2 fine-tuned BGE-small encoder + dual FAISS/BM25 index",
    ),
    "classifier_transformer": ArtifactBundle(
        key="classifier_transformer",
        gdrive_file_id="1EZMEmAI16wbhANoHW9A9yDWJ25D8wJqh",
        size_mb=399.0,
        target_subdir="classifier",
        marker_relpath="model.safetensors",
        description="Stage 1 fine-tuned DeBERTa-v3-small multi-task classifier",
    ),
    "llm_ollama": ArtifactBundle(
        key="llm_ollama",
        gdrive_file_id="REPLACE_WITH_GDRIVE_ID_LLM_OLLAMA",
        size_mb=3258.4,
        target_subdir="llm",
        marker_relpath="draft-qwen25-3b-q8_0.gguf",
        description="Stage 3 fine-tuned Qwen2.5-3B Q8_0 GGUF, Modelfile, and LoRA adapter",
    ),
}

PROFILES: dict[str, list[str]] = {
    "core": ["classifier_baseline", "retriever"],
    "no_llm": ["classifier_baseline", "retriever", "classifier_transformer"],
    "full": ["classifier_baseline", "retriever", "classifier_transformer", "llm_ollama"],
}


def _extract_bundle(zip_path: Path, artifacts_dir: Path, bundle: ArtifactBundle) -> Path:
    """Extract a zip archive cleanly into artifacts/<target_subdir>, normalizing nesting."""
    target_dir = artifacts_dir / bundle.target_subdir
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_dir = Path(tmp_str)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_dir)

        candidate = tmp_dir
        nested_artifacts = tmp_dir / "artifacts" / bundle.target_subdir
        nested_sub = tmp_dir / bundle.target_subdir
        if nested_artifacts.exists():
            candidate = nested_artifacts
        elif nested_sub.exists():
            candidate = nested_sub
        else:
            children = [p for p in tmp_dir.iterdir() if p.is_dir() and p.name != "__MACOSX"]
            if len(children) == 1 and (children[0] / bundle.marker_relpath).exists():
                candidate = children[0]

        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(candidate, target_dir, dirs_exist_ok=True)

    marker = target_dir / bundle.marker_relpath
    if not marker.exists():
        raise RuntimeError(
            f"Extraction finished for '{bundle.key}', but expected file is missing: {marker}"
        )
    return target_dir


def _download_one(bundle: ArtifactBundle, artifacts_dir: Path, force: bool) -> None:
    """Download and extract a single artifact bundle via gdown."""
    marker = artifacts_dir / bundle.target_subdir / bundle.marker_relpath
    if marker.exists() and not force:
        print(f"  [skip] {bundle.key} already present at {marker}")
        return

    if bundle.gdrive_file_id.startswith("REPLACE_WITH_"):
        raise ValueError(
            f"Google Drive file ID for '{bundle.key}' is not configured yet. "
            f"Update ARTIFACT_BUNDLES['{bundle.key}'].gdrive_file_id in "
            "scripts/download_artifacts.py."
        )

    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "gdown is required. Run with: "
            "uv run --with gdown python scripts/download_artifacts.py --profile core"
        ) from exc

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_str:
        zip_path = Path(tmp_str) / f"{bundle.key}.zip"
        print(f"  [download] {bundle.key} (~{bundle.size_mb:.1f} MB)...")
        out = gdown.download(id=bundle.gdrive_file_id, output=str(zip_path), quiet=False)
        if not out or not zip_path.exists():
            raise RuntimeError(
                f"Failed to download '{bundle.key}' from Google Drive (id={bundle.gdrive_file_id})."
            )
        target_dir = _extract_bundle(zip_path, artifacts_dir, bundle)
        print(f"  [ok] extracted to {target_dir}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for artifact downloads."""
    parser = argparse.ArgumentParser(
        description="Download pre-trained DRAFT model artifacts from Google Drive."
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES.keys()),
        default="core",
        help="core (~142 MB), no_llm (~541 MB), or full (~3.8 GB).",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        choices=sorted(ARTIFACT_BUNDLES.keys()),
        default=None,
        help="Optional explicit list of artifact bundles to download instead of --profile.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Destination directory for extracted artifacts (default: artifacts).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite existing artifacts.",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip interactive download size confirmation prompt.",
    )
    return parser.parse_args()


def main() -> None:
    """Resolve requested bundles, show size warning, and download artifacts."""
    args = parse_args()
    keys = args.components if args.components else PROFILES[args.profile]
    bundles = [ARTIFACT_BUNDLES[key] for key in keys]
    total_mb = sum(b.size_mb for b in bundles)

    print("Selected DRAFT artifact bundles:")
    for b in bundles:
        print(f"  - {b.key:24s} ~{b.size_mb:7.1f} MB  ({b.description})")
    print(f"Total estimated download size: ~{total_mb:.1f} MB (~{total_mb / 1024.0:.2f} GB)")

    if not args.yes:
        reply = input("Proceed with download? [y/N]: ").strip().lower()
        if reply not in {"y", "yes"}:
            print("Download cancelled.")
            return

    for bundle in bundles:
        _download_one(bundle, args.artifacts_dir, force=args.force)

    print("\nNext steps:")
    print("  1. Base LLM (recommended for policy grounding, ~1.9 GB):")
    print("       ollama pull qwen2.5:3b")
    if "llm_ollama" in keys:
        print("  2. Register fine-tuned GGUF model in Ollama:")
        print("       ollama create draft-qwen25-3b -f artifacts/llm/Modelfile")
    print("  3. Run CLI demo:")
    print(
        '       uv run draft generate-draft --subject "Duplicate charge" '
        '--body "We noticed a duplicate charge on our invoice issued 10 days ago." '
        "--classifier-backend baseline --generator-backend ollama "
        "--ollama-model qwen2.5:3b --few-shot"
    )


if __name__ == "__main__":
    main()
