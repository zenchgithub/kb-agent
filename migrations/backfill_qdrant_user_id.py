import argparse
import sys
from pathlib import Path

from qdrant_client import models

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_qdrant_client


def source_filter(source: str | None) -> models.Filter | None:
    if not source:
        return None
    return models.Filter(
        must=[
            models.FieldCondition(
                key="source",
                match=models.MatchValue(value=source),
            )
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Backfill user_id onto existing Qdrant points that do not have it."
    )
    parser.add_argument("--user-id", required=True, help="Supabase auth.users.id to assign")
    parser.add_argument("--collection", default="nas_docs")
    parser.add_argument("--source", help="Optional exact payload source, for example I-485.pdf")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    qc = get_qdrant_client()
    next_offset = None
    scanned = 0
    updated = 0

    while True:
        points, next_offset = qc.scroll(
            collection_name=args.collection,
            scroll_filter=source_filter(args.source),
            limit=256,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        scanned += len(points)

        point_ids = [
            point.id
            for point in points
            if not (point.payload or {}).get("user_id")
        ]
        if point_ids:
            updated += len(point_ids)
            if not args.dry_run:
                qc.set_payload(
                    collection_name=args.collection,
                    payload={"user_id": str(args.user_id)},
                    points=point_ids,
                )

        if next_offset is None:
            break

    action = "Would update" if args.dry_run else "Updated"
    print(f"{action} {updated} of {scanned} scanned points in {args.collection}")


if __name__ == "__main__":
    main()
