# Librerías
import logging
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from typing import List

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

    def add(self, documents: List[Document]) -> None:
        """
        Añade nuevos documentos al índice vectorial.
        
        Args:
            documents (List[Document]): Lista de Document de LangChain
        """
        # Si el atributo del índice vectorial está vacío, se crea a
        # partir de los documentos de Lnagchain
        if self.db is None:
            self.db = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
        
        # Si no, se añaden los nuevos documentos
        else:
            self.db.add_documents(documents)

        # Se guarda el índice en local
        self.db.save_local(self.path)
        logger.info(f"Índice actualizado con {len(documents)} chunks.")

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
    
    def search(self, embedding, k: int = 5) -> List[Document]:
        """
        Busca los documentos más similares a un embedding dado.
        
        Returns:
            List[Document]: Documentos recuperados (con page_content y
            metadata)
        """
        # Si el atributo del índice vectorial está vacío, se eleva error
        if self.db is None:
            raise ValueError("Vector store no inicializado")

        # Se buscan los documentos de langchain más similares a la consulta
        docs = self.db.similarity_search_by_vector(embedding, k=k)

        # Devuelve dichos documentos de Langchain
        return docs