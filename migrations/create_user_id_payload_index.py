import sys
from pathlib import Path

from qdrant_client import models

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_qdrant_client


COLLECTION_NAME = "nas_docs"


def main():
    qc = get_qdrant_client()
    for field_name, field_schema in (
        ("user_id", models.PayloadSchemaType.KEYWORD),
        ("isPublic", models.PayloadSchemaType.BOOL),
    ):
        try:
            qc.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=field_schema,
            )
            print(f"Created payload index: {COLLECTION_NAME}.{field_name}")
        except Exception as exc:
            print(f"Skipped payload index {COLLECTION_NAME}.{field_name}: {exc}")


if __name__ == "__main__":
    main()
