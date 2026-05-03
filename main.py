"""
Executable entry point for the Advanced RAG project.

Use this file to run only the module/workflow you need:
  - interactive generation app
  - source-aware indexing check
  - route debugging
  - prompt preview
  - query preprocessing
  - legacy indexing demo
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


BASE_DIR = Path(__file__).parent
DEFAULT_PDF_FOLDER = BASE_DIR / "rag_docs"


def _resolve_pdf_folder(path_value: str | None) -> Path:
    if not path_value:
        return DEFAULT_PDF_FOLDER
    path = Path(path_value)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def _build_router(pdf_folder: Path, k: int):
    from advanced_rag_indexing_2 import build_advanced_retrieval_index
    from advanced_rag_router import MultiSourceRouter

    pdf_folder.mkdir(exist_ok=True)
    pdf_files = sorted(pdf_folder.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {pdf_folder}. Add PDFs there before running this command."
        )

    print(f"PDF folder : {pdf_folder.resolve()}")
    print(f"PDF files  : {[path.name for path in pdf_files]}")

    index_bundle = build_advanced_retrieval_index(k=k, pdf_folder=pdf_folder)
    router = MultiSourceRouter(
        indexed_sources=index_bundle["indexed_sources"],
        global_retriever=index_bundle["global_retriever"],
        default_k=k,
    )
    return router, index_bundle


def run_interactive(args: argparse.Namespace) -> None:
    # advanced_generation_rag.py is the existing full REPL. Running it through
    # run_path preserves its current behavior without importing it for other commands.
    runpy.run_path(str(BASE_DIR / "advanced_generation_rag.py"), run_name="__main__")


def run_index(args: argparse.Namespace) -> None:
    pdf_folder = _resolve_pdf_folder(args.pdf_folder)
    _, index_bundle = _build_router(pdf_folder, args.k)
    print("\nIndex summary")
    print(f"  Sources : {len(index_bundle['indexed_sources'])}")
    print(f"  Chunks  : {len(index_bundle['all_splits'])}")


def run_route(args: argparse.Namespace) -> None:
    pdf_folder = _resolve_pdf_folder(args.pdf_folder)
    router, _ = _build_router(pdf_folder, args.k)
    routed = router.route(args.query, k=args.k)

    print("\nRoute debug")
    print(f"  Raw query       : {routed.processed_query.raw_query}")
    print(f"  Retrieval query : {routed.processed_query.retrieval_query}")
    print(f"  Search mode     : {routed.processed_query.search_mode}")
    print(f"  Filters         : {routed.processed_query.filters or '{}'}")
    print(f"  Route           : {routed.route_name}")
    print(f"  Sources         : {routed.selected_sources or ['global']}")
    print(f"  Issues          : {routed.processed_query.issues or ['none']}")
    print(f"  Variants        : {routed.processed_query.variants}")
    print("\nRetrieved chunks")
    for index, doc in enumerate(routed.documents, start=1):
        filename = doc.metadata.get("filename", "?")
        page = doc.metadata.get("page", "?")
        source_id = doc.metadata.get("source_id", "?")
        snippet = doc.page_content[:180].replace("\n", " ")
        print(f"  [{index}] {filename} | Page {page} | Source {source_id}")
        print(f"      {snippet!r}")


def run_preview(args: argparse.Namespace) -> None:
    from IndexingDocs_for_rag import build_prompt, format_docs

    pdf_folder = _resolve_pdf_folder(args.pdf_folder)
    router, _ = _build_router(pdf_folder, args.k)
    routed = router.route(args.query, k=args.k)
    prompt = build_prompt(args.prompt)
    rendered = prompt.format(
        context=format_docs(routed.documents),
        question=routed.processed_query.retrieval_query,
    )

    print("\nRendered prompt preview")
    print(rendered[: args.max_chars])
    if len(rendered) > args.max_chars:
        print("...")


def run_process_query(args: argparse.Namespace) -> None:
    from advanced_rag_query_processing import process_user_query

    processed = process_user_query(args.query)
    print("Processed query")
    print(f"  Raw query        : {processed.raw_query}")
    print(f"  Normalized query : {processed.normalized_query}")
    print(f"  Retrieval query  : {processed.retrieval_query}")
    print(f"  Search mode      : {processed.search_mode}")
    print(f"  Filters          : {processed.filters or '{}'}")
    print(f"  Variants         : {processed.variants}")
    print(f"  Issues           : {processed.issues or ['none']}")


def run_legacy_index(args: argparse.Namespace) -> None:
    runpy.run_path(str(BASE_DIR / "IndexingDocs_for_rag.py"), run_name="__main__")


def run_menu(args: argparse.Namespace) -> None:
    commands = {
        "1": ("interactive", "Run the full advanced RAG chat app"),
        "2": ("index", "Build source-aware indexes and show a summary"),
        "3": ("route", "Debug routing for one query"),
        "4": ("preview", "Render the prompt for one query"),
        "5": ("process-query", "Show query preprocessing only"),
        "6": ("legacy-index", "Run the older indexing demo"),
    }

    print("\nAdvanced RAG main executable")
    for key, (command, description) in commands.items():
        print(f"  {key}. {command:<13} - {description}")

    choice = input("\nSelect option [1-6]: ").strip()
    selected = commands.get(choice)
    if not selected:
        print("Invalid option.")
        return

    command = selected[0]
    if command in {"route", "preview", "process-query"}:
        query = input("Query: ").strip()
        if not query:
            print("Query is required.")
            return
        sys.argv = [sys.argv[0], command, query]
    else:
        sys.argv = [sys.argv[0], command]

    main()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Advanced RAG modules according to the current requirement."
    )
    subparsers = parser.add_subparsers(dest="command")

    interactive = subparsers.add_parser("interactive", help="Run the full RAG REPL.")
    interactive.set_defaults(func=run_interactive)

    index = subparsers.add_parser("index", help="Build source-aware indexes only.")
    index.add_argument("--pdf-folder", default=None, help="PDF folder. Default: rag_docs")
    index.add_argument("--k", type=int, default=6, help="Retriever top-k value.")
    index.set_defaults(func=run_index)

    route = subparsers.add_parser("route", help="Show source routing for a query.")
    route.add_argument("query", help="Query to route.")
    route.add_argument("--pdf-folder", default=None, help="PDF folder. Default: rag_docs")
    route.add_argument("--k", type=int, default=6, help="Retriever top-k value.")
    route.set_defaults(func=run_route)

    preview = subparsers.add_parser("preview", help="Preview the generated prompt.")
    preview.add_argument("query", help="Query to preview.")
    preview.add_argument(
        "--prompt",
        default="default",
        choices=["default", "concise", "detailed", "bullet"],
        help="Prompt style.",
    )
    preview.add_argument("--pdf-folder", default=None, help="PDF folder. Default: rag_docs")
    preview.add_argument("--k", type=int, default=6, help="Retriever top-k value.")
    preview.add_argument("--max-chars", type=int, default=2000, help="Preview character limit.")
    preview.set_defaults(func=run_preview)

    process_query = subparsers.add_parser(
        "process-query", help="Show query normalization, filters, and variants."
    )
    process_query.add_argument("query", help="Query to process.")
    process_query.set_defaults(func=run_process_query)

    legacy_index = subparsers.add_parser("legacy-index", help="Run IndexingDocs_for_rag.py.")
    legacy_index.set_defaults(func=run_legacy_index)

    menu = subparsers.add_parser("menu", help="Open a simple interactive command menu.")
    menu.set_defaults(func=run_menu)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not hasattr(args, "func"):
        args = parser.parse_args(["menu"])

    try:
        args.func(args)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
