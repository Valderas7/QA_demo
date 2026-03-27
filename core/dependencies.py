# Librerías
from functools import lru_cache
from services.ingest import IngestService
from services.chunking import ChunkService
from services.embedding import EmbeddingService
from services.vector_store import VectorStoreService
from services.reranker import Reranker
from services.llm_service import LLMService


class Services:
    """
    Clase contenedora de servicios. Permite inicializar y acceder a los
    servicios de ingesta, chunking, embedding, vector store, reranking y LLM
    desde un único punto, facilitando la gestión de dependencias y la
    reutilización de instancias a lo largo de la aplicación.
    """
    def __init__(self) -> None:
        """
        Inicializa todos los servicios necesarios para el pipeline RAG.
        """
        self.ingest = IngestService()
        self.chunk = ChunkService()
        self.embedding = EmbeddingService()
        self.vectorstore = VectorStoreService(
            embeddings=self.embedding.embeddings
        )
        self.reranker = Reranker()
        self.llm = LLMService()


@lru_cache
def get_services() -> Services:
    """
    Función de dependencia para obtener una instancia de Services. Se cachea
    para asegurar que se reutilice la misma instancia a lo largo de la
    aplicación, evitando la creación de múltiples instancias de los servicios

    Returns:
        Services: Instancia de la clase Services con todos los servicios
        inicializados.
    """
    return Services()