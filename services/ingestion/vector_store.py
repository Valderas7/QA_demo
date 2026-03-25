# Librerías
import logging
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from typing import List, Dict

# Logger del módulo
logger = logging.getLogger(__name__)


class VectorStoreService:
    """
    Servicio de almacenamiento vectorial basado en FAISS.

    Permite construir un índice vectorial a partir de chunks de texto y
    realizar recuperación semántica (retrieval) mediante embeddings.
    """
    def __init__(self, embeddings, path: str = "faiss_index"):
        """
        Inicializa el servicio de vector store.

        Args:
            embeddings: Servicio de embeddings compatible con LangChain.
            path (str): Ruta donde se guarda/carga el índice FAISS.
        """
        self.embeddings = embeddings.get()
        self.path = path
        self.db = self._load_or_create()

    def _load_or_create(self) -> FAISS:
        """
        Carga el índice FAISS desde disco si existe.
        Si no existe, inicializa el vector store vacío.

        Returns:
            FAISS | None: Índice vectorial cargado o None si no existe.
        """
        # Se intenta...
        try:

            # Cargar el índice desde local
            db = FAISS.load_local(
                self.path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("FAISS cargado desde disco.")

            # Se devuelve el índice vectorial
            return db

        # Si ocurre excepción, no se devuelve nada
        except Exception:
            logger.info("No existe índice, creando uno nuevo.")
            return None

    def add(self, texts: List[str], metadatas: List[Dict]) -> None:
        """
        Añade nuevos documentos al índice vectorial.

        Si el índice no existe, lo crea desde cero.
        Si ya existe, añade los nuevos embeddings de forma incremental.

        Args:
            texts (List[str]): Lista de textos (chunks).
            metadatas (List[Dict]): Metadata asociada a cada chunk
            (ej: page, source, chunk_id)
        """
        # Si el atributo está vacío se crea el índice
        if self.db is None:
            self.db = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
        
        # Si no está vacío, se añaden los nuevos vectores al índice
        else:
            self.db.add_texts(
                texts=texts,
                metadatas=metadatas
            )

        # Se guarda el índice en local
        self.db.save_local(self.path)
        logger.info(f"Índice actualizado con {len(texts)} chunks.")

    def get_retriever(self, k: int = 5) -> VectorStoreRetriever:
        """
        Devuelve un retriever configurado para búsqueda semántica.

        Args:
            k (int): Número de documentos relevantes a recuperar.

        Returns:
            VectorStoreRetriever: Retriever configurado con top-k.
        """
        if self.db is None:
            raise ValueError("Vector store no inicializado")

        # Se recuperan los Top K candidatos
        return self.db.as_retriever(search_kwargs={"k": k})