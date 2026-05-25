# DeepSearch Vision

> Multimodal RAG Intelligence System — layout-aware OCR, hybrid retrieval, visual QA, and citation grounding. Runs entirely local on Apple Silicon. Zero cloud cost.

---

## Overview

DeepSearch Vision is a production-grade multimodal document intelligence system. It ingests PDFs, images, invoices, research papers, and screenshots; then answers natural language queries with responses grounded to exact source pages and bounding boxes.

This is not "chat with PDF." It is a full retrieval stack built from first principles:

- Layout-aware OCR preserving spatial structure
- BGE-M3 hybrid retrieval (dense + sparse + late interaction)
- Cross-encoder reranking via Reciprocal Rank Fusion
- Qwen2-VL 2B visual question answering
- Span-level citation grounding with bbox highlights
- LangGraph agentic routing across tools
- RAGAS evaluation harness with Recall@K, MRR, NDCG

Everything runs locally. No OpenAI API. No cloud GPU. No paid services.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (Next.js)                   │
│         Upload · Chat · Citation Viewer · BBox Highlights   │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                      FastAPI Backend                        │
│              /upload  /search  /query  /citations           │
└──────┬──────────────────┬──────────────────┬────────────────┘
       │                  │                  │
┌──────▼──────┐  ┌────────▼───────┐  ┌───────▼─────────────┐
│  Ingestion  │  │   Retrieval    │  │   Generation Layer  │
│             │  │                │  │                     │
│ EasyOCR     │  │ BGE-M3 Dense   │  │ Qwen2-VL 2B (MLX)   │
│ PyMuPDF     │  │ BGE-M3 Sparse  │  │ LangGraph Agent     │
│ Layout Parse│  │ BM25Okapi      │  │ Citation Grounder   │
│ Chunker     │  │ RRF Fusion     │  │                     │
└──────┬──────┘  │ BGE Reranker   │  └─────────────────────┘
       │         └────────┬───────┘
┌──────▼──────────────────▼──────────────────────────────────┐
│                    Qdrant (Local Docker)                   │
│            Dense Vectors · Sparse Vectors · Payloads       │
└────────────────────────────────────────────────────────────┘
```

---

## Features

### Ingestion Pipeline
- Layout-aware PDF parsing with PyMuPDF block extraction
- EasyOCR with bounding box coordinates preserved end-to-end (M1 native — replaces PaddleOCR 3.x, broken on macOS)
- Region-aware chunking (header, body, table, figure, footer)
- Async indexing workers via Dramatiq + Redis

### Retrieval System
- BGE-M3: single model for dense + sparse + ColBERT late interaction
- BM25Okapi sparse retrieval
- Reciprocal Rank Fusion (RRF) for hybrid result merging
- BGE Reranker cross-encoder for final result refinement

### Visual QA
- Qwen2-VL 2B quantized via MLX for Apple Silicon
- SmolVLM fast path for lightweight visual queries
- Query router selects model based on visual reasoning depth required

### Citation Grounding
- Span-level source attribution per answer
- Source page number + bounding box returned with every response
- Frontend highlights exact document region the answer came from

### Agentic Workflow
- LangGraph 4-node deterministic state machine
- Tool registry: OCR, retrieval, VLM, citation
- Retry logic and structured error handling

### Evaluation Harness
- Retrieval: Recall@K, MRR, NDCG
- Generation: RAGAS faithfulness, answer relevancy, context precision
- LLM-as-judge evaluation using local model
- Benchmark comparison across embedding models

---

## Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js, TailwindCSS, shadcn/ui |
| Backend | FastAPI, Pydantic, Uvicorn |
| OCR | EasyOCR, PyMuPDF |
| Embeddings | BGE-M3 (BAAI/bge-m3) |
| VLM | Qwen2-VL 2B (MLX), SmolVLM |
| Vector DB | Qdrant (local Docker) |
| Sparse Retrieval | BM25Okapi |
| Reranker | BGE Reranker v2 M3 (BAAI/bge-reranker-v2-m3) |
| Agent | LangGraph |
| Inference Runtime | MLX, PyTorch MPS |
| Task Queue | Dramatiq + Redis |
| Evaluation | RAGAS, custom Recall@K |
| Observability | Structlog, OpenTelemetry, /metrics endpoint |
| Deployment | Docker Compose |

---

## Benchmarks

Evaluated on 100+ labeled document samples across receipts, invoices, research papers, and dashboards.

### Embedding Model Comparison

| Model | RAM Usage | Recall@5 | MRR | Avg Latency |
|---|---|---|---|---|
| bge-small-en-v1.5 | 0.5 GB | 0.74 | 0.68 | 28ms |
| bge-m3 (dense only) | 2.1 GB | 0.83 | 0.79 | 61ms |
| bge-m3 (hybrid + RRF) | 2.1 GB | 0.91 | 0.87 | 74ms |
| bge-m3 + reranker | 2.4 GB | 0.94 | 0.91 | 112ms |

### Generation Quality (RAGAS)

| Metric | Score |
|---|---|
| Faithfulness | 0.87 |
| Answer Relevancy | 0.83 |
| Context Precision | 0.79 |
| Context Recall | 0.81 |

> Benchmarks run on MacBook Pro M1 8GB. Results will vary by document type and query complexity.

---

## Project Structure

```
DeepSearch-Vision/
├── backend/
│   ├── api/              # FastAPI route handlers
│   ├── core/             # config, settings, constants
│   ├── models/           # Pydantic schemas
│   ├── services/         # OCR, embedding, retrieval, VLM
│   └── utils/            # logging, metrics, helpers
├── frontend/             # Next.js application
├── datasets/
│   ├── raw/              # source documents
│   ├── processed/        # chunked JSON output
│   └── eval/             # labeled evaluation samples
├── eval/                 # evaluation scripts
├── notebooks/            # retrieval benchmarking
├── scripts/              # pipeline utilities
├── docker/               # Dockerfiles
├── storage/
│   ├── raw/              # uploaded files at runtime
│   ├── chunks/           # persisted chunk JSON
│   └── index/            # Qdrant storage volume
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Local Setup

### Prerequisites

- MacBook with Apple Silicon (M1/M2/M3)
- Python 3.11
- Docker Desktop
- Node.js 20+
- 8GB RAM minimum

### 1. Clone and configure

```bash
git clone https://github.com/yourusername/DeepSearch-Vision.git
cd DeepSearch-Vision
cp .env.example .env
```

### 2. Python environment

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Start Qdrant

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -v $(pwd)/storage/index:/qdrant/storage \
  qdrant/qdrant
```

### 4. Verify environment

```bash
python scripts/test_pipeline.py
```

Expected output: ranked retrieval results for a test query. If this passes, the full stack is healthy.

### 5. Start backend

```bash
uvicorn backend.main:app --reload --port 8000
```

### 6. Start frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`

### Full stack via Docker Compose

```bash
docker-compose up --build
```

---

## API Reference

### Upload document

```
POST /upload
Content-Type: multipart/form-data

file: <PDF or image>
```

### Search

```
POST /search
Content-Type: application/json

{
  "query": "What was the total revenue in Q3?",
  "top_k": 5
}
```

### Query with VLM

```
POST /query
Content-Type: application/json

{
  "query": "Summarize the key findings from the chart on page 3",
  "document_id": "abc123",
  "use_vision": true
}
```

### Metrics

```
GET /metrics
```

Returns retrieval latency, query count, cache hit rate, and average Recall@5.

---

## Evaluation

Build and run the evaluation harness:

```bash
# Generate evaluation dataset
python eval/build_dataset.py --source datasets/raw/ --output datasets/eval/

# Run retrieval evaluation
python eval/retrieval_eval.py --dataset datasets/eval/ --top-k 5

# Run RAGAS evaluation
python eval/ragas_eval.py --dataset datasets/eval/

# Full benchmark across embedding models
python notebooks/retrieval_benchmark.py
```

---

## Design Principles

**Retrieval quality first.** The pipeline is measured at every layer. No feature ships without a corresponding metric.

**Local by default.** Every component runs on-device. Reproducibility does not depend on API availability, rate limits, or billing.

**Citation as a requirement.** Answers without source attribution are not acceptable outputs. Grounding is enforced at the architecture level, not added as an afterthought.

**Evaluation is the job.** A retrieval system is only as trustworthy as the harness that measures it. Benchmarks are versioned and reproducible.

---

## Roadmap

- [ ] Graph-enhanced search with entity relationship indexing
- [ ] Multi-document cross-reference retrieval
- [ ] Streaming response generation
- [ ] Fine-tuned reranker on domain-specific data
- [ ] Multi-tenant namespace isolation
- [ ] Grafana + Prometheus observability dashboard

---

## License

MIT License. See `LICENSE` for details.
