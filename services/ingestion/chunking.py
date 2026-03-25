# Librerías
import logging
import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict

# Se obtiene el logger para este módulo
logger = logging.getLogger(__name__)


class ChunkService:
    """
    Servicio de chunking. Divide texto en chunks utilizando un splitter
    recursivo basado en caracteres, pero con conteo de tokens real
    mediante tiktoken."""
    def __init__(self, model_name: str = "gpt-4o-mini"):
        """
        Inicializa el encoder de tokens según el modelo seleccionado.

        Args:
            model_name (str): Modelo usado para definir tokenización.
        """
        self.model_name = model_name
        self.encoder = tiktoken.encoding_for_model(model_name)

    def _get_length_function(self):
        """
        Devuelve una función de conteo de tokens para el splitter.

        Returns:
            Callable: función que recibe texto y devuelve número de tokens.
        """
        return lambda text: len(self.encoder.encode(text))

    def chunk_text(
        self,
        pages: List[Dict],
        chunk_size: int = 512,
        chunk_overlap: int = 100
    ) -> List[Document]:
        """
        Divide una lista de páginas en chunks de tamaño fijo basados en
        tokens.

        Este método aplica tokenización usando tiktoken y genera ventanas
        deslizantes (sliding window) con overlap para mantener contexto
        entre chunks.

        Cada chunk mantiene metadata de origen para trazabilidad
        (página, fuente y ID del chunk).

        Args:
            pages (List[Dict]): Lista de páginas con estructura:
            {
                "text": str,
                "page": int,
                "source": str
            }
            chunk_size (int): Número máximo de tokens por chunk.
            overlap (int): Número de tokens compartidos entre chunks
            consecutivos.

        Returns:
            List[Document]: Lista de Document de LangChain con:
            - page_content: texto del chunk
            - metadata: {"page": int, "source": str, "chunk_id": int}
        """
        # Si no hay páginas se devuelve lista vacía
        if not pages:
            return []

        # Se crea el splitter recursivo
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self._get_length_function(),
            separators=[
                "\n\n",     # Párrafos
                "\n",       # Saltos de línea
                ". ",       # Oraciones
                "! ",
                "? ",
                " ",        # Palabras
                ""          # Caracteres (último recurso)
            ],
            keep_separator=True,
            strip_whitespace=True,
        )

        # Lista vacía e ID de chunk inicial
        chunks = []
        chunk_id = 0

        # Para cada página...
        for page in pages:
            
            # Se obtiene el campo 'text' del diccionario
            text = page.get("text", "").strip()

            # Si no hay texto se continúa
            if not text:
                continue

            # Se aplica el splitter recursivo al texto de la página
            split_texts = text_splitter.split_text(text)

            # Para cada chunk...
            for split_text in split_texts:

                # Se añade a la lista un Documento con el texto como contenido
                # y los metadatos de página, fuente e ID
                chunks.append(
                    Document(
                        page_content=split_text,
                        metadata={
                            "page": page.get("page"),
                            "source": page.get("source"),
                            "chunk_id": chunk_id
                        }
                    )
                )

                # Se aumenta en uno el contador de chunks
                chunk_id += 1

        # Se devuelve la lista de documentos
        logger.info(
            f"Generados {len(chunks)} chunks recursivos de {len(pages)} "
            "páginas."
        )
        return chunks