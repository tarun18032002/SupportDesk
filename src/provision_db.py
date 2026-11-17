"""Create or recreate the Qdrant collection for SupportDesk.

Examples:
    uv run python src/provision_db.py
    uv run python src/provision_db.py --recreate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (str(ROOT), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from config import (
    COLLECTION_NAME,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    db_api_key,
    db_url,
)
from src.vector_store import get_client, ensure_collection_exists, recreate_collection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Provision the SupportDesk Qdrant collection"
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the collection (required after embedding dim changes)",
    )
    return parser.parse_args()


def main() -> None:
    if not db_url:
        raise RuntimeError("Set DB_URL in .env before provisioning")

    args = parse_args()
    client = get_client(
        url=db_url,
        api_key=db_api_key,
    )

    try:
        print(
            f"Provider={EMBEDDING_PROVIDER} model={EMBEDDING_MODEL} "
            f"dim={EMBEDDING_DIMENSION} collection={COLLECTION_NAME}"
        )
        if args.recreate:
            recreate_collection(
                client=client,
                collection=COLLECTION_NAME,
                vector_size=EMBEDDING_DIMENSION,
            )
        else:
            ensure_collection_exists(
                client=client,
                collection=COLLECTION_NAME,
                vector_size=EMBEDDING_DIMENSION,
            )
    finally:
        client.close()


if __name__ == "__main__":
    main()
