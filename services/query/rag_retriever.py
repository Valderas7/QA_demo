# Librerías
from langchain_core.documents import Document
from typing import List


class Retriever:
    """
    Componente de recuperación (retrieval) en un pipeline RAG.

    Este servicio transforma una consulta en lenguaje natural en un
    embedding vectorial y realiza una búsqueda semántica en la base
    vectorial para obtener los chunks más relevantes.
    """

    def __init__(self, embedder, vectorstore):
        """
        Inicializa el retriever con los servicios necesarios.

        Args:
            embedder:
                Servicio encargado de generar embeddings (ej: SentenceTransformer).

            vectorstore:
                Base de datos vectorial (ej: FAISS) que almacena embeddings
                junto con su metadata asociada.
        """
        self.embedder = embedder
        self.vs = vectorstore

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Recupera los chunks más relevantes para una consulta dada.

        Convierte la consulta en un embedding y realiza una búsqueda
        de similitud en la base vectorial para obtener los `top_k`
        resultados más cercanos semánticamente.

        Args:
            query (str): Pregunta o consulta en lenguaje natural.
            top_k (int): Número de resultados más relevantes a recuperar.

        Returns:
            List[Document]: Lista de documentos de Langchain
        """
        embedding_model = self.embedder.get()
        q_emb = embedding_model.embed_query(query)

        # Buscamos en el vectorstore
        return self.vs.search(q_emb, top_k)