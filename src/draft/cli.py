"""Command-line interface for DRAFT pipeline stages."""

from pathlib import Path
from typing import Annotated

import typer

from draft.config import load_config
from draft.data.acquisition import DataAcquisitionService

app = typer.Typer(help="DRAFT: Dual-source RAG and classification CLI.")


@app.callback()
def main() -> None:
    """CLI entrypoint for DRAFT."""


@app.command("prepare-data")
def prepare_data(
    config: Annotated[
        Path, typer.Option(help="Path to YAML config file.")
    ] = Path("configs/default.yaml"),
    force_download: Annotated[
        bool, typer.Option(help="Re-download raw Hugging Face dataset.")
    ] = False,
    overwrite_org_docs: Annotated[
        bool, typer.Option(help="Regenerate default organization docs.")
    ] = False,
) -> None:
    """Download, clean, split datasets and write the EDA summary report."""
    app_config = load_config(config)
    summary = DataAcquisitionService(app_config).run(
        force_download=force_download,
        overwrite_org_docs=overwrite_org_docs,
    )

    stats = summary["cleaning_stats"]
    splits = summary["split_sizes"]
    typer.echo("Data preparation completed:")
    typer.echo(
        f"  Raw rows: {stats['raw_rows']} -> EN: {stats['after_language_filter']} "
        f"-> Valid labels: {stats['after_label_filter']} "
        f"-> Deduplicated: {stats['after_deduplication']}"
    )
    typer.echo(
        f"  Splits: train={splits['train']}, val={splits['val']}, test={splits['test']}"
    )
    typer.echo(f"  Org docs: {summary['org_docs_total']}")
    for label, path in summary["output_paths"].items():
        typer.echo(f"  [{label}] {path}")
