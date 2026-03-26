# Librerías
from services.ingestion.embedding import EmbeddingService
from services.ingestion.vector_store import VectorStoreService

# Se inicializa una sola instancia del almacén de vectores como singleton
embedder = EmbeddingService()
vectorstore = VectorStoreService(embedder)