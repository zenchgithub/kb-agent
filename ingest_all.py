import yaml
from ingest import ingest

with open("collections.yaml") as f:
    config = yaml.safe_load(f)

for coll in config.get("collections", []):
    name = coll.get("name")
    for fp in coll.get("files", []):
        print(f"Ingesting {fp} into collection '{name}'")
        ingest(fp, name)
