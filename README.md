# Advanced RAG Pipeline

This project is a local Retrieval-Augmented Generation (RAG) pipeline built with LangChain, HuggingFace embeddings, an in-memory vector store, source-aware routing, and Ollama for answer generation.

The active pipeline indexes PDFs from `rag_docs/`, builds one retriever per PDF plus a global retriever, routes each user query to the most relevant source, retrieves matching chunks, and sends the retrieved context to a local LLM.

## Project Introduction

This project demonstrates an end-to-end advanced RAG workflow for asking questions over local PDF documents. The completed work includes document loading, text cleaning, chunking, embedding generation, vector indexing, source-aware routing, query preprocessing, multiple retrieval strategies, prompt selection, and local answer generation.

In simple terms, the project turns research papers and lecture PDFs into a searchable knowledge base. When a user asks a question, the system finds the most relevant document chunks and uses those chunks as context so the LLM can answer from the indexed files instead of relying only on its pre-trained knowledge.

Main libraries and their role in this project:

| Library / Tool | Use In This Project |
| --- | --- |
| LangChain | Connects the RAG workflow pieces: prompts, retrievers, chains, document objects, output parsing, and vector-store access. |
| Ollama | Runs the local LLM used for query rewriting and final answer generation. The active advanced pipeline uses `llama3.2:1b`. |
| HuggingFace Embeddings | Converts document chunks and user queries into vectors for semantic search. The project uses `sentence-transformers/all-MiniLM-L6-v2`. |
| InMemoryVectorStore | Stores embeddings in memory during runtime and returns similar chunks for each query. |
| pdfplumber | Extracts text page by page from PDF files before chunking and indexing. |
| Tavily Search | Optional web search tool for `web_search` and `hybrid` retrieval modes when `TAVILY_API_KEY` is configured. |

## Active Entry Point

Run the main executable dispatcher with:

```bash
python main.py
```

You can also run a specific workflow directly:

```bash
python main.py interactive
python main.py index
python main.py route "file:attention explain multi-head attention"
python main.py preview "summarize diffusion models" --prompt bullet
python main.py process-query "  compare rag and llm??  "
python main.py legacy-index
```

The older direct advanced RAG application entry point still works:

```bash
python advanced_generation_rag.py
```

The script starts an interactive prompt where you can ask questions, switch retrieval modes, preview prompts, debug routing, stream output, and run batch queries.

## Document Folder

The advanced pipeline reads PDFs from:

```text
rag_docs/
```

Current documents include papers and lecture material such as:

- `Attention_is_All_You_Need.pdf`
- `MachineLearning-Lecture01.pdf`
- `BERT_Pre-training of Deep Bidirectional Transformers for Language Understanding.pdf`
- `Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.pdf`
- `Denoising Diffusion Probabilistic Models.pdf`
- `Generative Adversarial Networks.pdf`
- `An Introduction to Variational Autoencoders.pdf`

To add more knowledge to the RAG system, place another PDF in `rag_docs/` and restart `advanced_generation_rag.py`.

## Project Files

```text
.
|-- main.py                           # Executable dispatcher for running workflows by requirement
|-- advanced_generation_rag.py        # Main interactive advanced RAG pipeline
|-- advanced_rag_indexing_2.py        # Active source-aware PDF indexing module
|-- advanced_rag_router.py            # Routes queries to source-specific or global retrievers
|-- advanced_rag_query_processing.py  # Normalizes queries, expands abbreviations, extracts filters
|-- IndexingDocs_for_rag.py           # Shared indexing, embedding, prompt, tracker, and vector helpers
|-- advanced_rag_indexing.py          # Earlier fixed-PDF indexing version
|-- test.py                           # Scratch/test code for RAG fusion logic
|-- rag_docs/                         # PDF corpus used by the active advanced pipeline
|-- hf_doc_change_tracker.json        # HuggingFace document tracking metadata
|-- hf_record_manager.db              # SQLite record manager database for indexing demos
|-- langgraph_agentic_rag_architecture.svg
|-- langgraph_rag_extension_architecture.svg
```

## How The Pipeline Executes

When you run `python advanced_generation_rag.py`, the workflow is:

1. `advanced_generation_rag.py` sets `PDF_FOLDER` to `rag_docs/`.
2. It checks that the folder contains PDFs.
3. It calls `build_advanced_retrieval_index()` from `advanced_rag_indexing_2.py`.
4. `advanced_rag_indexing_2.py` discovers PDFs, creates source configs, loads each PDF page, cleans text, splits text into chunks, embeds chunks, and builds retrievers.
5. The returned index bundle contains:
   - `indexed_sources`: one `IndexedSource` object per PDF.
   - `global_retriever`: a retriever over all chunks from all PDFs.
   - `all_splits`: every generated chunk.
   - `embeddings`: the HuggingFace embedding model.
6. `advanced_generation_rag.py` creates a `MultiSourceRouter` from `advanced_rag_router.py`.
7. It builds a prompt with `build_prompt()` from `IndexingDocs_for_rag.py`.
8. It initializes `ChatOllama(model="llama3.2:1b", temperature=0)`.
9. It builds the generation chain for the selected retrieval mode.
10. For each user query, the chain rewrites or expands the query if needed, retrieves context through the router, formats retrieved chunks, sends the prompt to Ollama, and prints the answer.

## Module Responsibilities

### `advanced_generation_rag.py`

This is the main application file.

It imports:

- `build_prompt` and `format_docs` from `IndexingDocs_for_rag.py`
- `build_advanced_retrieval_index` from `advanced_rag_indexing_2.py`
- `MultiSourceRouter` from `advanced_rag_router.py`
- `ChatOllama`, `ChatPromptTemplate`, `StrOutputParser`, and LangChain chain utilities
- optional `TavilySearchResults` for web search modes

Main responsibilities:

- Starts the interactive RAG REPL.
- Builds the source-aware index from `rag_docs/`.
- Creates the router and generation chain.
- Supports retrieval modes:
  - `rewrite_rag`
  - `multiquery_rag`
  - `rag_fusion`
  - `web_search`
  - `hybrid`
- Supports prompt styles:
  - `default`
  - `concise`
  - `detailed`
  - `bullet`
- Provides debug commands such as `route:`, `preview:`, and `rewrite:`.

Important note: the executable code is at module level, not inside an `if __name__ == "__main__"` guard. Run this file directly; importing it from another script will also start the pipeline.

### `advanced_rag_indexing_2.py`

This is the active indexing module used by `advanced_generation_rag.py`.

It imports shared helpers from `IndexingDocs_for_rag.py`:

- `build_embeddings`
- `build_vectorstore`
- `clean_text`
- `get_or_create_doc_id`
- `load_tracker`
- `update_tracker`

Main responsibilities:

- Recursively discovers PDF files under the selected folder.
- Skips non-project folders such as `.git`, `venv`, `.venv`, `node_modules`, and `__pycache__`.
- Builds one `SourceConfig` per discovered PDF.
- Chooses chunk settings from filename heuristics:
  - attention/transformer: chunk size `700`, overlap `100`
  - lecture: chunk size `500`, overlap `50`
  - survey/report: chunk size `600`, overlap `80`
  - default: chunk size `500`, overlap `50`
- Loads PDF text page by page with `pdfplumber`.
- Cleans extracted text.
- Splits pages into retrievable chunks.
- Adds `chunk_id`, `doc_id`, `filename`, `page`, `source_id`, and `source_name` metadata.
- Builds:
  - a source-specific in-memory vector store for each PDF
  - a global in-memory vector store across all PDFs
- Updates `hf_doc_change_tracker.json` with hash, timestamp, document ID, and chunk count.

### `advanced_rag_router.py`

This module decides where retrieval should happen.

It imports:

- `ProcessedQuery`
- `process_user_query`

from `advanced_rag_query_processing.py`.

Main responsibilities:

- Converts the raw query into a structured processed query.
- Selects source retrievers using explicit filters first.
- Falls back to soft keyword overlap between the query and source filenames.
- Uses the global retriever when no source-specific match is found.
- Retrieves using query variants.
- Deduplicates chunks by `chunk_id`.

Routing examples:

```text
file:attention explain multi-head attention
source:lecture explain gradient descent
type:pdf summarize transformer architecture
```

### `advanced_rag_query_processing.py`

This module prepares raw user input for retrieval.

Main responsibilities:

- Normalizes whitespace and repeated punctuation.
- Expands common technical abbreviations:
  - `rag` -> `retrieval augmented generation`
  - `llm` -> `large language model`
  - `nlp` -> `natural language processing`
  - `ml` -> `machine learning`
  - `ai` -> `artificial intelligence`
  - `db` -> `database`
  - `pg` -> `postgresql`
- Extracts inline filters:
  - `file:<name>`
  - `doc:<name>`
  - `document:<name>`
  - `source:<name>`
  - `type:<source_type>`
- Builds query variants for better recall.
- Infers query mode:
  - `general`
  - `comparison`
  - `summary`
  - `grounded_lookup`
- Flags query quality issues such as very short input, extra whitespace, or repeated characters.

### `IndexingDocs_for_rag.py`

This is the shared utility and demonstration script.

It provides helpers imported by the advanced modules:

- `clean_text(text)`: removes PDF/OCR artifacts and normalizes whitespace.
- `compute_file_hash(filepath)`: creates an MD5 hash for change tracking.
- `load_tracker()` and `save_tracker()`: read/write indexing metadata.
- `get_or_create_doc_id(filepath, tracker)`: keeps stable document IDs across runs.
- `update_tracker(filepath, tracker, num_chunks, doc_id)`: records indexing results.
- `build_embeddings()`: creates HuggingFace embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
- `build_vectorstore(embeddings)`: creates an `InMemoryVectorStore`.
- `format_docs(docs)`: formats retrieved chunks for the prompt.
- `build_prompt(prompt_type)`: returns prompt templates used by the generator.
- `build_retriever(vectorstore, pg_vectorstore, k)`: chooses PGVector when available, otherwise in-memory retrieval.

When executed directly with:

```bash
python IndexingDocs_for_rag.py
```

it runs a longer indexing demo over the hard-coded root-level PDFs listed in `PDF_FILES`, including document tracking, PDF loading, chunking, embedding, in-memory vector search, optional PGVector storage, and a simple interactive RAG loop.

### `advanced_rag_indexing.py`

This is an earlier version of the source-aware indexing module.

It uses the hard-coded `PDF_FILES` list from `IndexingDocs_for_rag.py`, so it only indexes the configured root-level PDFs. The active advanced pipeline currently uses `advanced_rag_indexing_2.py` instead because that version can discover PDFs from `rag_docs/`.

### `test.py`

This file contains scratch/test logic related to Reciprocal Rank Fusion. It is not part of the main execution path for `advanced_generation_rag.py`.

## Retrieval Modes

The interactive app starts in `rewrite_rag` mode. Type `mode` in the REPL to switch modes.

| Mode | Purpose |
| --- | --- |
| `rewrite_rag` | Rewrites the user question once, then retrieves from local PDF chunks. |
| `multiquery_rag` | Uses the LLM to generate 3 retrieval queries, retrieves for each, then deduplicates chunks. |
| `rag_fusion` | Generates 4 retrieval queries, retrieves for each, and reranks chunks with Reciprocal Rank Fusion. |
| `web_search` | Rewrites the question for web search and queries Tavily. Requires `TAVILY_API_KEY`. |
| `hybrid` | Combines local PDF context with Tavily web search context. |

## Query-To-Answer Flow

For a normal query in `rewrite_rag` mode:

```text
User question
  -> LLM query rewrite
  -> MultiSourceRouter.route()
  -> process_user_query()
  -> choose source-specific retriever or global retriever
  -> retrieve matching chunks
  -> deduplicate chunks
  -> format_docs()
  -> build_prompt()
  -> ChatOllama
  -> final answer
```

For `multiquery_rag`:

```text
User question
  -> LLM generates 3 search queries
  -> route each query
  -> merge retrieved chunks
  -> deduplicate top chunks
  -> generate answer
```

For `rag_fusion`:

```text
User question
  -> LLM generates 4 search queries
  -> route each query
  -> keep ranked result lists
  -> apply Reciprocal Rank Fusion
  -> select top fused chunks
  -> generate answer
```

For `hybrid`:

```text
User question
  -> local RAG rewrite and retrieval
  -> web-search rewrite and Tavily retrieval
  -> combine local and web context
  -> generate answer
```

## Interactive Commands

After running `advanced_generation_rag.py`, use these commands:

| Command | Description |
| --- | --- |
| `<query>` | Ask a normal question using the active prompt and retrieval mode. |
| `prompt` | Switch prompt style. |
| `mode` | Switch retrieval mode. |
| `preview:<query>` | Show the rendered prompt before generation. |
| `rewrite:<query>` | Show RAG, web, and multi-query rewrites. |
| `route:<query>` | Show query preprocessing, selected route, and retrieved chunks. |
| `stream:<query>` | Stream answer tokens. |
| `batch:<q1>|<q2>` | Run multiple questions in one batch. |
| `exit` | Stop the REPL. |

## Prompt Types

Type `prompt` in the REPL to switch.

| Prompt | Behavior |
| --- | --- |
| `default` | Standard grounded answer using retrieved context. |
| `concise` | Short 1-2 sentence answer. |
| `detailed` | More complete answer with document/page citations. |
| `bullet` | Bullet-point answer. |

## Setup

Install Python dependencies:

```bash
pip install langchain langchain-community langchain-core langchain-ollama langchain-huggingface langchain-text-splitters pdfplumber sentence-transformers
```

Install and run Ollama, then pull the model used by the advanced pipeline:

```bash
ollama pull llama3.2:1b
```

Optional web search support:

```bash
pip install langchain-community
```

Set your Tavily API key before using `web_search` or `hybrid` mode:

```bash
set TAVILY_API_KEY=your_key_here
```

PowerShell:

```powershell
$env:TAVILY_API_KEY="your_key_here"
```

## Running The Advanced RAG Pipeline

1. Put PDFs in `rag_docs/`.
2. Start Ollama.
3. Run:

```bash
python advanced_generation_rag.py
```

4. Ask questions, for example:

```text
What is self-attention?
file:attention explain multi-head attention
compare VAEs and GANs
source:lecture summarize supervised learning
route:what is diffusion?
preview:file:bert explain masked language modeling
```

## Tracking And Storage

The active advanced pipeline uses:

- `hf_doc_change_tracker.json` for document hash, `doc_id`, indexing timestamp, and chunk count.
- `InMemoryVectorStore` for vector storage during the current process.

Because the active vector stores are in memory, the PDFs are reloaded and re-embedded when you restart the advanced pipeline. The tracker still records document identity and indexing metadata.

`IndexingDocs_for_rag.py` also contains optional PGVector support, but the main `advanced_generation_rag.py` path currently uses in-memory vector stores.
