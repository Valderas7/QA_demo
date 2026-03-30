# Librerías
import logging
import pickle
import re
from core.constants import Constants
from langchain_core.documents import Document
from pathlib import Path
from rank_bm25 import BM25Okapi
from typing import List

# Logger del módulo
logger = logging.getLogger(__name__)


class LexicalSearchService:
    """
    Servicio de recuperación basado en BM25. Permite indexar documentos y
    realizar búsquedas."""
    def __init__(self, index_path: str = "data/bm25_index.pkl") -> None:
        """
        Inicializa el servicio de recuperación BM25.

        Args:
            index_path (str): Ruta donde se guardará o cargará el índice BM25
            persistente.
        """
        self.bm25 = None
        self.documents: List[Document] = []
        self.tokenized_corpus: List[List[str]] = []
        self.index_path = Path(index_path)
        self.sources: set[str] = set()
        self._load_index()

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokeniza un texto en una lista de palabras.

        Args:
            text (str): Texto a tokenizar.

        Returns:
            List[str]: Lista de palabras tokenizadas.
        """
        return [
            w for w in re.findall(r"\b\w+\b", text.lower())
            if w not in Constants.STOPWORDS
        ]

    def _load_index(self) -> None:
        """
        Carga el índice BM25 desde un archivo persistente si existe. Si no
        existe, se inicializa un índice vacío.
        """
        # Si el archivo de índice BM25 existe...
        if self.index_path.exists():

            # Se intenta...
            try:

                # Cargar el índice BM25 desde el archivo utilizando pickle y
                # actualizar los atributos del servicio con los datos cargados
                with open(self.index_path, "rb") as f:
                    data = pickle.load(f)
                    self.documents = data["documents"]
                    self.tokenized_corpus = data["tokenized_corpus"]
                    self.sources = data.get("sources", set())
                    self.bm25 = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None
                logger.info(
                    f"Índice BM25 cargado desde disco: {len(self.documents)} "
                    "documentos."
                )

            # Si hay excepción al cargar el índice, el atributo sigue vacío
            except Exception as e:
                logger.error(f"Error al cargar el índice BM25: {e}")
        
        # Si en cambio no existe el índice, se loggea
        else:
            logger.info(
                "No existe índice BM25 en disco. Se inicializará al "
                "añadir documentos."
            )

    def _save_index(self) -> None:
        """Guarda el índice BM25 en disco."""

        # Se intenta...
        try:

            # Se crea el directorio para guardar el índice si no existe
            self.index_path.parent.mkdir(parents=True, exist_ok=True)

            # Se abre el archivo de índice BM25 en modo escritura binaria y se
            # guarda el índice utilizando pickle, guardando tanto la lista de
            # documentos como el corpus tokenizado necesario para reconstruir
            # el índice BM25
            with open(self.index_path, "wb") as f:
                pickle.dump({
                    "documents": self.documents,
                    "tokenized_corpus": self.tokenized_corpus,
                    "sources": self.sources
                }, f)

        # Si hay excepción al guardar el índice, se loggea el error
        except Exception:
            logger.exception("Error guardando el índice BM25 en disco.")

    def has_source(self, source: str) -> bool:
        """
        Comprueba rápidamente si un documento ya está indexado en BM25.

        Args:
            source (str): Nombre del documento o fuente a comprobar.

        Returns:
            bool: True si ya existe al menos un chunk con esa fuente, False si
            no existe ningún chunk con esa fuente.
        """
        # Se comprueba si el nombre del documento (source) ya está en el
        # conjunto de fuentes indexadas.
        exists = source in self.sources

        # Se muestra un log indicando si el documento ya existe o es nuevo
        # según el resultado de la comprobación
        if exists:
            logger.info(f"El documento '{source}' ya está indexado en BM25.")
        else:
            logger.info(f"El documento '{source}' es nuevo para BM25.")

        # Se retorna el resultado de la comprobación de existencia del
        # documento
        return exists

    def add_chunks(self, chunks: List[Document]) -> None:
        """
        Agrega chunks al índice BM25. Se tokenizan los textos de los
        chunks y se construye el índice BM25 con el corpus actualizado.

        Args:
            chunks (List[Document]): Lista de chunks a agregar al
            índice.
        """
        # Si no se proporcionan chunks, no se hace nada
        if not chunks:
            return
        
        # Se crea una lista para almacenar los nuevos chunks que se van a
        # agregar al índice y un conjunto para almacenar los nombres de los
        # documentos fuente de esos nuevos chunks
        new_docs: List[Document] = []
        new_sources: set[str] = set()
        
        # Para cada chunk de la lista de chunks proporcionada...
        for doc in chunks:

            # Se comprueba el metadato "source" del chunk para obtener el
            # nombre del documento fuente
            source = doc.metadata.get("source")

            # Si el nombre del documento fuente existe y no está ya indexado
            # en BM25, se añade el chunk a la lista de nuevos chunks a agregar
            # y se añade el nombre del documento fuente al conjunto de nuevas
            # fuentes
            if source and source not in self.sources:
                new_docs.append(doc)
                new_sources.add(source)

        # Si no hay nuevos chunks para agregar, se loggea y se retorna
        # sin hacer nada
        if not new_docs:
            logger.info("Todos los chunks ya estaban indexados en BM25.")
            return
        
        # Se añaden los nuevos chunks a la lista de chunks existente
        self.documents.extend(new_docs)

        # Se tokenizan los textos de los nuevos chunks
        new_tokenized = [self._tokenize(doc.page_content) for doc in new_docs]

        # Se extiende el corpus con los nuevos chunks tokenizados
        self.tokenized_corpus.extend(new_tokenized)

        # Se actualiza el set de documentos fuente indexados (una sola vez
        # por documento)
        self.sources.update(new_sources)

        # Se construye el índice BM25 con el corpus actualizado
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # Se guarda el índice BM25 actualizado en disco para persistencia
        self._save_index()
        logger.info(
            f"BM25 actualizado: +{len(new_docs)} chunks | "
            f"Total chunks: {len(self.documents)}."
        )

    def search(self, query: str, k: int = 20) -> List[Document]:
        """
        Realiza una búsqueda en el índice BM25 para recuperar los chunks
        más relevantes para la consulta dada.

        Args:
            query (str): Consulta de texto para buscar chunks relevantes.
            k (int): Número de resultados más relevantes a devolver.
            
        Returns:
            List[Document]: Lista de chunks más relevantes según la
            consulta.
        """
        # Si no hay índice BM25 inicializado, se devuelve una lista vacía
        if not self.bm25:
            return []

        # Se tokeniza la consulta para realizar la búsqueda en el índice BM25
        tokenized_query = self._tokenize(query)

        # Se obtienen los puntajes de relevancia para cada documento en el
        # índice
        scores = self.bm25.get_scores(tokenized_query)

        # Se ordenan los documentos por puntaje de relevancia
        ranked_docs = sorted(
            zip(self.documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        # Se devuelven los k documentos más relevantes según el puntaje de
        # relevancia calculado por BM25
        return [doc for doc, _ in ranked_docs[:k]]