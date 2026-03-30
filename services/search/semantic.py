# Librerías
import logging
import threading
from core.exceptions import VectorStoreNotInitializedError
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from typing import List

# Logger del módulo
logger = logging.getLogger(__name__)


class SemanticSearchService:
    """
    Servicio de almacenamiento vectorial basado en Qdrant (local persistente).

    Permite construir un índice vectorial a partir de chunks de texto y
    realizar recuperación semántica (retrieval) mediante embeddings.
    """
    def __init__(
        self,
        embeddings: Embeddings,
        path: str = "qdrant_db"
    ) -> None:
        """
        Inicializa el servicio de base de datos vectorial.

        Args:
            embeddings (Embeddings): Instancia del servicio de embeddings
            para generar vectores.
            path (str): Ruta donde se almacenará la base de datos vectorial
            de Qdrant.
        """
        # Inicializa los atributos del servicio
        self.embeddings = embeddings
        self.path = Path(path)
        self.collection_name = "qa_demo"
        self.client = QdrantClient(path=self.path)
        self.db: QdrantVectorStore | None = self._load_vectorstore()
        self.lock = threading.Lock()

    def _load_vectorstore(self) -> QdrantVectorStore | None:
        """
        Carga la colección de Qdrant si existe.
        Si no existe, devuelve None y se creará una nueva colección al
        añadir documentos.

        Returns:
            QdrantVectorStore | None: Instancia de QdrantVectorStore
            si se cargó correctamente, o None si no existe índice previo.
        """
        # Se intenta...
        try:
            
            # Se obtienen las colecciones existentes en Qdrant
            collections = self.client.get_collections().collections

            # Si alguna de las colecciones tiene el mismo nombre que la
            # colección que se quiere usar, se carga esa colección y se
            # devuelve la instancia de QdrantVectorStore para interactuar
            # con ella
            if any(coll.name == self.collection_name for coll in collections):
                logger.info(f"Colección '{self.collection_name}' cargada.")
                return QdrantVectorStore(
                    client=self.client,
                    collection_name=self.collection_name,
                    embedding=self.embeddings,
                )
        
            # Si no, no se retorna nada
            else:
                logger.info(f"Colección '{self.collection_name}' no existe.")
                return None

        # Si ocurre excepción, no se devuelve nada
        except Exception:
            logger.info(
                "No existe índice en Qdrant. Se creará uno nuevo al "
                "añadir documentos."
            )
            return None

    def add_documents(self, documents: List[Document]) -> None:
        """
        Añade nuevos documentos al índice vectorial de Qdrant. Si el índice
        no existe, se crea una nueva colección con la configuración adecuada
        para almacenar los vectores generados a partir de los documentos
        utilizando los embeddings.
        
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

            # Si el atributo 'db' es None, significa que no se ha inicializado
            # un índice
            if self.db is None:
                
                # Si la colección no existe en Qdrant...
                if not self.client.collection_exists(self.collection_name):
                    
                    # Se calcula el tamaño de los vectores a partir de los
                    # embeddings para configurar la colección correctamente
                    vector_size = (
                        len(self.embeddings.embed_query("test"))
                        )

                    # Se crea la colección en Qdrant con el nombre
                    # especificado y la configuración de vectores adecuada
                    # (tamaño y distancia de similitud)
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(
                            size=vector_size,
                            distance=Distance.COSINE,
                        ),
                    )

                # Se inicializa el atributo 'db' con una nueva instancia del
                # índice vectorial de Qdrant, apuntando a la colección recién
                # creada
                self.db = QdrantVectorStore(
                    client=self.client,
                    collection_name=self.collection_name,
                    embedding=self.embeddings,
                )
            
            # Se añaden los documentos al índice vectorial
            self.db.add_documents(documents)

        # Se muestra un log con el número de chunks añadidos y el total de
        # chunks en el índice después de la actualización
        total = self.client.count(self.collection_name).count
        logger.info(
            f"Qdrant actualizado: +{len(documents)} chunks | Total: {total}"
        )
    
    def similarity_search(self, query: str, k: int = 20) -> List[Document]:
        """
        Realiza una búsqueda semántica en el índice vectorial utilizando
        la consulta dada.

        Args:
            query (str): Consulta de texto para buscar documentos relevantes.
            k (int): Número de resultados más similares a devolver.

        Returns:
            List[Document]: Lista de Document de LangChain más relevantes
            según la consulta.
        """
        # Si no hay índice vectorial inicializado, se lanza una excepción
        if self.db is None:
            raise VectorStoreNotInitializedError(
                "El índice vectorial no está inicializado."
            )

        # Se realiza la búsqueda de similitud, devolviendo los k documentos
        # más relevantes para la consulta dada, basándose en los embeddings
        # de la consulta y los de los documentos en el índice
        return self.db.similarity_search(query, k=k)
