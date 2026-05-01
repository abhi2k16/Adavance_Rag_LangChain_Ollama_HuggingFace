def reciprocal_rank_fusion(results: list[list], k: int = 60, verbose: bool = False) -> list:
    scores: dict[tuple, float] = {}
    doc_map = {}
    per_query_contributions: dict[tuple, list[tuple[int, int, float]]] = {}  # doc_key -> [(query_idx, rank, score)]

    for query_idx, ranked_docs in enumerate(results):
        for rank, doc in enumerate(ranked_docs, start=1):
            doc_key = (
                doc.page_content,
                tuple(sorted(doc.metadata.items())) if getattr(doc, "metadata", None) else (),
            )
            contribution = 1.0 / (k + rank)
            scores[doc_key] = scores.get(doc_key, 0.0) + contribution
            doc_map[doc_key] = doc

            if doc_key not in per_query_contributions:
                per_query_contributions[doc_key] = []
            per_query_contributions[doc_key].append((query_idx + 1, rank, contribution))

    ranked_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    if verbose:
        print("\n" + "=" * 60)
        print("RAG FUSION — RRF SCORES")
        print(f"  k (smoothing constant) : {k}")
        print(f"  Formula                : score += 1 / (k + rank)")
        print(f"  Total unique docs      : {len(scores)}")
        print(f"  Total query lists      : {len(results)}")
        print("=" * 60)

        for position, (doc_key, total_score) in enumerate(ranked_docs, start=1):
            doc = doc_map[doc_key]
            filename  = doc.metadata.get("filename", "?")
            page      = doc.metadata.get("page", "?")
            source_id = doc.metadata.get("source_id", "?")
            snippet   = doc.page_content[:120].replace("\n", " ")
            contribs  = per_query_contributions[doc_key]

            print(f"\n  [{position:>2}] Total RRF Score : {total_score:.6f}")
            print(f"       File           : {filename}  |  Page {page}  |  {source_id}")
            print(f"       Snippet        : {snippet!r}...")
            print(f"       Appearances    : {len(contribs)} / {len(results)} query lists")
            for query_idx, rank, contrib in sorted(contribs):
                bar = "█" * max(1, int(contrib * 3000))
                print(f"         Query {query_idx}: rank #{rank:>2}  →  1/({k}+{rank}) = {contrib:.6f}  {bar}")

        print("\n" + "-" * 60)
        print(f"  Top 5 docs selected for context generation")
        print("=" * 60 + "\n")

    return [doc_map[doc_key] for doc_key, _ in ranked_docs]


def retrieve_rag_fusion_context(question: str) -> tuple[str, list, list[str]]:
    generated = rag_fusion_query_generator.invoke({"question": question}).strip()
    queries = parse_multi_queries(generated)
    if not queries:
        queries = [question]

    print("\n" + "=" * 60)
    print("RAG FUSION — GENERATED QUERIES")
    print("=" * 60)
    for i, q in enumerate(queries, start=1):
        print(f"  [{i}] {q}")
    print("=" * 60)

    ranked_results = []
    for i, query in enumerate(queries, start=1):
        routed = router.route(query, k=4)
        ranked_results.append(routed.documents)

        print(f"\n  Query [{i}] retrieved chunks:")
        for rank, doc in enumerate(routed.documents, start=1):
            filename  = doc.metadata.get("filename", "?")
            page      = doc.metadata.get("page", "?")
            snippet   = doc.page_content[:100].replace("\n", " ")
            print(f"    Rank #{rank} | {filename} p.{page} | {snippet!r}...")

    fused_docs = reciprocal_rank_fusion(ranked_results, k=60, verbose=True)[:5]
    return format_docs(fused_docs), fused_docs, queries