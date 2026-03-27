# Librerías
import logging
import threading
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from typing import List

# Logger del módulo
logger = logging.getLogger(__name__)


class VectorStoreService:
    """
    Servicio de almacenamiento vectorial basado en FAISS.

    Permite construir un índice vectorial a partir de chunks de texto y
    realizar recuperación semántica (retrieval) mediante embeddings.
    """
    def __init__(self, embeddings: Embeddings, path: str = "faiss_index"):
        """
        Inicializa el servicio de base de datos vectorial.

        Args:
            embeddings (Embeddings): Instancia de embeddings para generar
            vectores a partir de texto.
            path (str): Ruta donde se guarda/carga el índice FAISS.
        """
        self.embeddings = embeddings
        self.path = path
        self.db = self._load_or_create()
        self.lock = threading.Lock()

    def _load_or_create(self) -> FAISS:
        """
        Carga el índice FAISS desde disco si existe.
        Si no existe, inicializa el vector store vacío.

        Returns:
            FAISS | None: Índice vectorial cargado o None si no existe.
        """
        # Se intenta...
        try:

            # Cargar el índice desde local permitiendo deserialización
            # peligrosa (para evitar errores de versión)
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
        # Si no se proporcionan documentos, se muestra una advertencia y se
        # sale de la función sin hacer nada
        if not documents:
            logger.warning("No se proporcionaron documentos para añadir.")
            return

        # Se adquiere el lock para asegurar que solo un hilo pueda
        # modificar el índice a la vez, evitando problemas de concurrencia
        with self.lock:

            # Si el índice vectorial está vacío, se crea a partir de los chunks
            # proporcionados y los embeddings
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

        # Se recopilan cuantos chunks totales hay
        total_docs = len(self.db.index_to_docstore_id) if self.db else 0
        logger.info(
            f"Índice actualizado con {len(documents)} chunks nuevos. "
            f"Total: {total_docs} chunks."
        )
    
    def search(self, query: str, k: int = 5) -> List[Document]:
        """
        Realiza una búsqueda semántica en el índice vectorial.

        Args:
            query (str): Consulta de texto para buscar documentos relevantes.
            k (int): Número de resultados más similares a devolver.

        Returns:
            List[Document]: Lista de Document de LangChain más relevantes
            según la consulta.
        """
        # Si no hay índice vectorial inicializado, se lanza una excepción
        if self.db is None:
            raise ValueError("Vector store no inicializado.")

        # Se realiza la búsqueda de similitud utilizando la búsqueda por
        # similitud
        return self.db.similarity_search(query, k=k)
