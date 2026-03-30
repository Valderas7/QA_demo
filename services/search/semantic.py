# Librerías
import logging
import threading
from core.exceptions import VectorStoreNotInitializedError
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, Filter, FieldCondition, MatchValue
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
        path: str = "data/qdrant_db"
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
        self.db: QdrantVectorStore | None = None
        self.lock = threading.Lock()
        self.sources: set[str] = set()
        self._load_vectorstore_and_sources()

    def _load_vectorstore_and_sources(self) -> None:
        """
        Carga la base de datos vectorial de Qdrant y reconstruye el set de
        documentos fuente indexados. Si la colección no existe, se inicializa
        sin índice vectorial ni fuentes.
        """
        # Se intenta...
        try:

            # Si la colección no existe en Qdrant...
            if not self.client.collection_exists(self.collection_name):
                
                # No se devuelve nada
                logger.info("No existe colección Qdrant aún.")
                self.db = None
                return

            # Si en cambio, si existe, se inicializa el atributo 'db' con una
            # nueva instancia del índice vectorial de Qdrant, apuntando a la
            # colección existente
            self.db = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
            )
        
            # Se realiza una consulta de scroll en Qdrant para obtener un
            # punto del índice, incluyendo el campo 'source' de los vectores,
            # con el fin de reconstruir el set de documentos fuente indexados
            all_points = self.client.scroll(
                collection_name=self.collection_name,
                limit=1,
                with_payload=["source"]
            )[0]

            # Si se obtienen puntos del índice, se extraen los nombres de los
            # documentos fuente de los campos 'source' y se actualiza el set
            # de fuentes indexadas
            if all_points:
                self.sources = {
                    point.payload.get("source")
                    for point in all_points
                    if point.payload
                }
                logger.info(
                    f"Qdrant cargado: {len(self.sources)} documentos fuente "
                    "indexados."
                )
            
            else:
                logger.info("Colección Qdrant cargada pero está vacía.")

        # Si ocurre excepción, no se devuelve nada
        except Exception:
            logger.warning("Error al cargar la colección Qdrant")
            self.db = None
        
    def has_source(self, source: str) -> bool:
        """
        Comprueba si ya existe algún chunk de este documento (source)
        en Qdrant.

        Args:
            source (str): Nombre del documento o fuente a comprobar.

        Returns:
            bool: True si ya existe al menos un chunk con esa fuente,
            False si no existe ningún chunk con esa fuente.
        """
        # Si no hay índice vectorial inicializado o la colección no
        # existe en Qdrant, se asume que no existe el documento
        if self.db is None or not self.client.collection_exists(self.collection_name):
            return False

        # Se realiza una consulta de conteo en Qdrant para contar cuántos
        # vectores tienen un campo 'source' que coincida con el nombre del
        # documento
        result = self.client.count(
            collection_name=self.collection_name,
            count_filter=Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=source)
                    )
                ]
            )
        )

        # Si el conteo es mayor que 0, significa que ya existe al menos un
        # chunk con esa fuente, por lo que se retorna True.
        # Si el conteo es 0, se retorna False.
        exists = result.count > 0

        # Se muestra un log indicando si el documento ya existe o es nuevo
        # según el resultado de la comprobación
        if exists:
            logger.info(f"El documento '{source}' ya está indexado en Qdrant.")
        else:
            logger.info(f"El documento '{source}' es nuevo para Qdrant.")

        # Se retorna el resultado de la comprobación de existencia del
        # documento
        return exists

    def add_chunks(self, chunks: List[Document]) -> None:
        """
        Añade nuevos chunks al índice vectorial de Qdrant. Si el índice
        no existe, se crea una nueva colección con la configuración adecuada
        para almacenar los vectores generados a partir de los chunks
        utilizando los embeddings.
        
        Args:
            chunks (List[Document]): Lista de Document de LangChain
        """
        # Si no se proporcionan chunks, se muestra una advertencia y se
        # sale de la función sin hacer nada
        if not chunks:
            logger.warning("No se proporcionaron chunks para añadir.")
            return
        
        # Se crea una lista para almacenar los nuevos chunks que se van a
        # agregar al índice y un conjunto para almacenar los nombres de los
        # documentos fuente de esos nuevos chunks
        new_docs: List[Document] = []
        new_sources: set[str] = set()

        # Para cada chunk de la lista de chunks proporcionada...
        for chunk in chunks:

            # Se comprueba el metadato "source" del chunk para obtener el
            # nombre del documento fuente
            source = chunk.metadata.get("source")

            # Si el nombre del documento fuente existe y no está ya indexado
            # en Qdrant, se añade el chunk a la lista de nuevos chunks a
            # agregar
            if source and source not in self.sources:
                new_docs.append(chunk)
                new_sources.add(source)

        # Si no hay nuevos chunks para agregar, se loggea y se retorna
        # sin hacer nada
        if not new_docs:
            logger.info("Todos los chunks ya estaban indexados en Qdrant.")
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
            
            # Se añaden los nuevos chunks al índice vectorial
            self.db.add_documents(new_docs)

        # Se actualiza el set de documentos fuente indexados (una sola vez
        # por documento)
        self.sources.update(new_sources) 

        # Se muestra un log con el número de chunks añadidos y el total de
        # chunks en el índice después de la actualización
        total = self.client.count(self.collection_name).count
        logger.info(
            f"Qdrant actualizado: +{len(chunks)} chunks | "
            f"Total chunks: {total}."
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
