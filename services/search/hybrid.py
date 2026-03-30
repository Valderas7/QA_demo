# Librerías
from typing import List, Dict
from langchain_core.documents import Document
from services.search.semantic import SemanticSearchService
from services.search.lexical import LexicalSearchService

class HybridSearchService:
    """
    Servicio de recuperación híbrida que combina búsqueda semántica y léxica.
    Implementa una fusión de resultados mediante Reciprocal Rank Fusion (RRF)
    para integrar los resultados de ambos métodos de recuperación."""
    def __init__(
        self,
        semantic_search: SemanticSearchService,
        lexical_search: LexicalSearchService
    ) -> None:
        """
        Inicializa el servicio de recuperación híbrida con las instancias del
        vector store para búsqueda semántica y el servicio BM25 para búsqueda
        léxica.
        """
        self.semantic_search = semantic_search
        self.lexical_search = lexical_search

    def _rrf(
        self,
        results: List[List[Document]],
        k: int = 60,
        weights: List[float] | None = None
    ) -> List[Document]:
        """
        Aplica Reciprocal Rank Fusion (RRF) para fusionar los resultados de
        búsqueda semántica y léxica.

        Args:
            results (List[List[Document]]): Lista de listas de documentos
            resultantes de cada método de recuperación (semántica y léxica).
            k (int): Parámetro para el cálculo de RRF
            weights (List[float], optional): Lista de pesos para cada lista de
            resultados

        Returns:
            List[Document]: Lista de documentos fusionados y ordenados por
            relevancia.
        """
        # Diccionarios para almacenar scores acumulados y mapeo de documentos
        # por ID
        scores: Dict[str, float] = {}
        docs_map: Dict[str, Document] = {}

        # Si no se proporcionan pesos para las listas de resultados, se asigna
        # un peso por defecto de 0.7 para la búsqueda semántica y 0.3 para la
        # búsqueda léxica
        weights = weights or [0.7, 0.3]

        # Para cada par de peso y su correspondiente lista de resultados...
        for weight, result_list in zip(weights, results):

            # Para cada chunk en la lista de resultados...
            for rank, doc in enumerate(result_list):

                # Se obtiene un ID único para el documento, utilizando el ID
                # del chunk, o si no, un hash
                doc_id = doc.metadata.get("chunk_id") or str(hash(doc.page_content))

                # Se almacena el chunk en el mapeo de documentos por ID para
                # poder recuperarlo posteriormente
                docs_map[doc_id] = doc

                # Se calcula el score acumulado de RRF para el chunk, sumando
                # el peso multiplicado por el inverso del rango (rank)
                # del documento en la lista de resultados
                scores[doc_id] = scores.get(doc_id, 0) + weight * (1 / (k + rank))

        # Se ordenan los documentos por su score acumulado de RRF en orden
        # descendente, y en caso de empate, por su ID
        ranked = sorted(
            scores.items(),
            key=lambda x: (-x[1], x[0]),
        )

        # Se devuelve la lista de documentos ordenados por relevancia según el
        # score acumulado de RRF
        return [docs_map[doc_id] for doc_id, _ in ranked]

    def retrieve(self, query: str, k: int = 20) -> List[Document]:
        """
        Realiza una recuperación híbrida combinando búsqueda semántica y
        léxica, y devuelve los documentos más relevantes para la consulta
        dada.
        
        Args:
            query (str): Consulta de texto para buscar documentos relevantes.
            k (int): Número de resultados más relevantes a devolver.
            
        Returns:
            List[Document]: Lista de documentos más relevantes según la
            consulta, fusionados por RRF.
        """
        # Se realiza una búsqueda semántica en la base de datos vectorial
        # obteniendo los k chunks más relevantes para la consulta dada
        semantic_results = self.semantic_search.similarity_search(query, k=k)

        # Se realiza una búsqueda léxica en el índice BM25 obteniendo los k
        # documentos más relevantes para la consulta dada
        lexical_results = self.lexical_search.search(query, k=k)

        # Se fusionan los resultados de ambas búsquedas utilizando RRF para
        # obtener una lista ordenada de documentos más relevantes según la
        # consulta con un peso asignado a cada método de recuperación (0.7
        # para semántica y 0.3 para léxica)
        fused = self._rrf(
            [semantic_results, lexical_results],
            weights=[0.7, 0.3]
        )

        # Se devuelven los k documentos más relevantes tras la fusión de
        # resultados
        return fused[:k]