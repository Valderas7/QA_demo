# Librerías
import logging
import tiktoken
from functools import lru_cache
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

    @lru_cache(maxsize=10000)
    def _count_tokens(self, text: str) -> int:
        """
        Cuenta el número de tokens en un texto. Se cachea para mejorar
        rendimiento en textos repetidos, recordando hasta 10000 entradas.

        Args:
            text (str): Texto a tokenizar.

        Returns:
            int: Número de tokens en el texto.
        """
        return len(self.encoder.encode(text))
    
    def _create_splitter(
        self,
        chunk_size: int,
        chunk_overlap: int
    ) -> RecursiveCharacterTextSplitter:
        """
        Crea un splitter recursivo configurado para dividir texto en chunks
        basados en tokens, con un tamaño máximo y overlap definido.

        El splitter recursivo intenta dividir primero por párrafos, luego por
        líneas, oraciones, palabras y finalmente por caracteres.

        La función de longitud cuenta tokens reales en vez de caracteres,
        lo que permite un chunking más preciso para modelos de lenguaje.

        Args:
            chunk_size (int): Número máximo de tokens por chunk.
            chunk_overlap (int): Número de tokens compartidos entre chunks
            consecutivos.

        Returns:
            RecursiveCharacterTextSplitter: Configurado para tokenización
            real y separación jerárquica.
        """
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self._count_tokens,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "! ",
                "? ",
                " ",
                ""
            ],
            keep_separator=True,
            strip_whitespace=True,
        )

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
        # Si no hay lista de diccionarios con el texto del documento,
        # se devuelve una lista vacía
        if not pages:
            return []
        
        # Se crea el splitter recursivo
        splitter = self._create_splitter(chunk_size, chunk_overlap)

        # Lista vacía para almacenar los chunks generados
        chunks: List[Document] = []

        # Para cada diccionario en la lista de páginas...
        for page in pages:
            
            # Se obtiene el texto de la página del PDF
            text = page.get("text")

            # Si no hay texto en la página, se continúa con la siguiente
            # página
            if not text:
                continue

            # Se fragmenta el texto en chunks utilizando el splitter recursivo
            split_texts = splitter.split_text(text)

            # Para cada índice local y fragmento de texto...
            for local_id, split_text in enumerate(split_texts):
                
                # Si el chunk tiene menos de 20 tokens, se considera pequeño y
                # se omite para evitar generar chunks irrelevantes
                if self._count_tokens(split_text) < 20:
                    continue
                
                # Se genera un ID único para el chunk combinando la fuente,
                # el número de página y un ID local del chunk
                chunk_id = f"{page['source']}_p{page['page']}_c{local_id}"

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

        # Mensaje de logging con el número de chunks generados y el número de
        # páginas procesadas
        logger.info(
            f"Generados {len(chunks)} chunks recursivos de {len(pages)} "
            "páginas."
        )

        # Se devuelve la lista de Documentos con los chunks generados
        return chunks