# Librerías
from functools import lru_cache
from services.chunking import ChunkingService
from services.embedding import EmbeddingService
from services.search.hybrid import HybridSearchService
from services.ingestion import IngestionService
from services.search.lexical import LexicalSearchService
from services.llm_service import LLMService
from services.reranker import RerankerService
from services.search.semantic import SemanticSearchService


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
        self.ingestion = IngestionService()
        self.chunking = ChunkingService()
        self.embedding = EmbeddingService()
        self.semantic_search = SemanticSearchService(
            embeddings=self.embedding.embeddings
        )
        self.lexical_search = LexicalSearchService()
        self.hybrid_retriever = HybridSearchService(
            semantic_search=self.semantic_search,
            lexical_search=self.lexical_search
        )
        self.reranker = RerankerService()
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