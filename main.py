"""Main entry point for the Empório da Música AI Customer Support Agent."""

import argparse
import uvicorn

from src.api.cli import run_cli
from src.config.settings import settings
from src.db.ingestion import ingest_all_csvs


def main() -> None:
    """Parse CLI arguments and launch either the Interactive CLI or FastAPI Server."""
    parser = argparse.ArgumentParser(
        description="Empório da Música — AI Customer Support Agent (Artefact Challenge)"
    )
    parser.add_argument(
        "--mode",
        choices=["cli", "api"],
        default="cli",
        help="Execution mode: 'cli' for interactive terminal chat (default), or 'api' for FastAPI + Web UI server.",
    )
    parser.add_argument(
        "--session",
        default="cli_user_01",
        help="Session ID for conversation persistence in CLI mode.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.port,
        help="Port for the FastAPI server when running in 'api' mode.",
    )
    args = parser.parse_args()

    # Ensure database and CSV tables are initialized
    ingest_all_csvs(force_reload=False)

    if args.mode == "api":
        uvicorn.run("src.api.app:app", host="127.0.0.1", port=args.port, reload=False)
    else:
        run_cli(session_id=args.session)


if __name__ == "__main__":
    main()
