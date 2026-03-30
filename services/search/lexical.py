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
    def __init__(self, index_path: str = "bm25_index.pkl") -> None:
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
                    self.bm25 = BM25Okapi(self.tokenized_corpus)
                logger.info(
                    f"Índice BM25 cargado desde disco: {len(self.documents)} "
                    "documentos."
                )

            # Si hay excepción al cargar el índice, el atributo sigue vacío
            except Exception as e:
                logger.error(f"Error al cargar el índice BM25: {e}")
                self.bm25 = None
        
        # Si en cambio no existe el índice, se loggea
        else:
            logger.info(
                "No existe índice BM25 en disco. Se inicializará al "
                "agregar documentos."
            )
            self.bm25 = None

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
                }, f)

        # Si hay excepción al guardar el índice, se loggea el error
        except Exception:
            logger.exception("Error guardando el índice BM25 en disco.")

    def add_documents(self, documents: List[Document]) -> None:
        """
        Agrega documentos al índice BM25. Se tokenizan los textos de los
        documentos y se construye el índice BM25 con el corpus actualizado.

        Args:
            documents (List[Document]): Lista de documentos a agregar al
            índice.
        """
        # Si no se proporcionan documentos, no se hace nada
        if not documents:
            return
        
        # Se añaden los nuevos documentos a la lista de documentos existente
        self.documents.extend(documents)

        # Se tokenizan los textos de los nuevos documentos
        new_tokenized = [self._tokenize(doc.page_content) for doc in documents]

        # Se extiende el corpus tokenizado con los nuevos documentos
        # tokenizados
        self.tokenized_corpus.extend(new_tokenized)

        # Se construye el índice BM25 con el corpus tokenizado actualizado
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info(
            f"BM25 actualizado: +{len(documents)} docs | "
            f"Total: {len(self.documents)}."
        )

    def search(self, query: str, k: int = 20) -> List[Document]:
        """
        Realiza una búsqueda en el índice BM25 para recuperar los documentos
        más relevantes para la consulta dada.

        Args:
            query (str): Consulta de texto para buscar documentos relevantes.
            k (int): Número de resultados más relevantes a devolver.
            
        Returns:
            List[Document]: Lista de documentos más relevantes según la
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