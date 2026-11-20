from __future__ import annotations

import os
import urllib.request
from pathlib import Path

from .dataset import build_text_subset, file_hash

T2_REVISION = "921dd3af6e78d1ae7ee0368aa8d7eaee02c8f08e"
T2_REPOSITORY = "mteb/T2Retrieval"


def build_t2(output: str | Path, cache: str | Path, **options) -> Path:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("T2 import requires: pip install -e .[benchmark]") from exc
    cache_path = Path(cache).resolve() / "T2Retrieval" / T2_REVISION
    cache_path.mkdir(parents=True, exist_ok=True)
    tables, checksums = {}, {}
    for part in ("corpus", "queries", "data"):
        target = cache_path / f"{part}.parquet"
        if not target.exists():
            url = f"https://huggingface.co/datasets/{T2_REPOSITORY}/resolve/{T2_REVISION}/{part}/dev-00000-of-00001.parquet"
            print(f"Downloading {part} from {T2_REPOSITORY}@{T2_REVISION}", flush=True)
            temporary = target.with_suffix(".part")
            request = urllib.request.Request(url, headers={"User-Agent": "LocalMind-eval/0.3"})
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as handle:
                length = response.headers.get("Content-Length")
                downloaded = 0
                while block := response.read(1024 * 1024):
                    handle.write(block)
                    downloaded += len(block)
                if length and downloaded != int(length):
                    raise IOError("Incomplete download; cached final file was not created")
            parquet.read_schema(temporary)
            os.replace(temporary, target)
        checksums[part] = file_hash(target)
        tables[part] = parquet.read_table(target).to_pylist()
    documents = [{"id": str(row["_id"]), "text": row["text"]} for row in tables["corpus"]]
    queries = [{"id": str(row["_id"]), "text": row["text"]} for row in tables["queries"]]
    qrels = [{"query_id": str(row["query-id"]), "file_id": str(row["corpus-id"]),
              "relevance": int(row["score"])} for row in tables["data"]]
    return build_text_subset(documents, queries, qrels, output, source={
        "repository": T2_REPOSITORY, "revision": T2_REVISION, "original_split": "dev",
        "license": "apache-2.0", "sha256": checksums,
        "url": "https://huggingface.co/datasets/mteb/T2Retrieval",
    }, **options)
