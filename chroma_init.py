#!/usr/bin/env python3
import argparse
import os
import glob
import shutil

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

def load_markdown_files(folder: str):
    paths = sorted(glob.glob(os.path.join(folder, "*.md")))
    docs = []
    metadatas = []
    for i, p in enumerate(paths):
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        docs.append(text)
        metadatas.append({
            "filename": os.path.basename(p),
            "path": os.path.abspath(p),
            "n_chars": len(text),
        })
    ids = [str(i) for i in range(len(docs))]
    return ids, docs, metadatas

def main():
    parser = argparse.ArgumentParser(description="Build local ChromaDB from wcag_techniques/*.md")
    parser.add_argument("--folder", default="wcag_techniques", help="Folder containing .md files")
    parser.add_argument("--persist", default="./wcag_chroma", help="Chroma persist directory")
    parser.add_argument("--collection", default="wcag_docs", help="Collection name")
    parser.add_argument("--reset", action="store_true", help="Delete and rebuild persist dir")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model")
    args = parser.parse_args()

    if args.reset and os.path.isdir(args.persist):
        print(f"[reset] removing existing persist dir: {args.persist}")
        shutil.rmtree(args.persist)

    # Local, persistent Chroma client
    client = chromadb.PersistentClient(
        path=args.persist,
        settings=Settings(anonymized_telemetry=False)
    )

    # Local embedding function (runs on CPU; no internet required)
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=args.model)

    # Create or get the collection
    collection = client.get_or_create_collection(
        name=args.collection,
        embedding_function=embed_fn,
        metadata={"source": "wcag_techniques"}
    )

    # Load all .md files
    ids, docs, metadatas = load_markdown_files(args.folder)
    if not ids:
        print(f"No .md files found in {args.folder}")
        return

    # Clear previous contents if rebuilding same collection without --reset
    # (Optional) Uncomment to always replace contents:
    # existing = collection.count()
    # if existing:
    #     print(f"[info] Clearing existing {existing} items in collection '{args.collection}'")
    #     collection.delete(where={})  # delete all

    print(f"Indexing {len(ids)} documents from {args.folder} -> {args.collection}")
    # Add in batches (safer for large sets)
    B = 256
    for start in range(0, len(ids), B):
        end = start + B
        collection.add(
            ids=ids[start:end],
            documents=docs[start:end],
            metadatas=metadatas[start:end],
        )

    print(f"Done. Persisted to: {os.path.abspath(args.persist)}")
    print(f"Collection '{args.collection}' count: {collection.count()}")

    # Quick sanity check: peek a couple of docs
    sample = collection.get(ids=["0"]) if len(ids) > 0 else None
    if sample and sample.get("documents"):
        print("\n[peek] ID=0")
        print("filename:", sample["metadatas"][0]["filename"])
        print("chars:", sample["metadatas"][0]["n_chars"])
        print("preview:", (sample["documents"][0][:200] + "...").replace("\n", " "))

if __name__ == "__main__":
    main()
