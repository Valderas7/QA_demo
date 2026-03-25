import logging
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from typing import List

# Logger del módulo
logger = logging.getLogger(__name__)


class Reranker:
    """
    Servicio de re-ranking basado en cross-encoder.

    Reordena documentos recuperados mediante similarity search utilizando
    un modelo cross-encoder, que evalúa la relevancia semántica de cada
    par (query, documento).
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ) -> None:
        """
        Inicializa el modelo de re-ranking.

        Args:
            model_name (str): Nombre del modelo cross-encoder.
        """
        self.model = CrossEncoder(model_name)
        logger.info(f"Reranker inicializado con modelo: '{model_name}'.")

    def rerank(
        self,
        query: str,
        docs: List[Document],
        top_k: int = 3
    ) -> List[Document]:
        """
        Reordena documentos según relevancia semántica respecto a la query.

        Args:
            query (str): Pregunta del usuario.
            docs (List[Document]): Documentos recuperados inicialmente.
            top_k (int): Número de documentos a devolver tras re-ranking.

        Returns:
            List[Document]: Lista de documentos reordenados por relevancia.
        """
        # Si no hay documentos se devuelve lista vacía
        if not docs:
            logger.warning("Reranker recibió lista vacía de documentos.")
            return []

        # Construcción de pares (query, documento)
        pairs = [[query, doc.page_content] for doc in docs]

        # Predicción de scores de relevancia
        scores = self.model.predict(pairs)

        # Ordenar documentos por score descendente
        ranked = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        # Se seleccionan los Top K documentos
        top_docs = [doc for doc, _ in ranked[:top_k]]
        logger.info(f"Top-{top_k} documentos seleccionados tras re-ranking.")

        # Se devuelven dichos documentos
        return top_docs