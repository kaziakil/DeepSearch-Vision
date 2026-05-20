# Run Qdrant locally (persistent + reproducible)

docker run -d \
  --name qdrant \
  --restart unless-stopped \
  -p 6333:6333 \
  -p 6334:6334 \
  -v $(pwd)/storage/qdrant:/qdrant/storage \
  qdrant/qdrant:v1.11.3

curl http://localhost:6333/health
# Expected: {"title":"qdrant - vector search engine","version":"..."}
Qdrant verified running on localhost:6333

