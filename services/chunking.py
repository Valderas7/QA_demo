# Librerías
import logging
import uuid
import tiktoken
from functools import lru_cache
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict

# Se obtiene el logger para este módulo
logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def get_encoder(model_name: str):
    """
    Obtiene un encoder de tokens para el modelo especificado. Se cachea para
    evitar crear múltiples instancias del encoder, lo que puede ser costoso en
    términos de memoria y rendimiento.

    Args:
        model_name (str): Nombre del modelo para el cual se desea obtener el
        encoder.
    """
    return tiktoken.encoding_for_model(model_name)


class ChunkingService:
    """
    Servicio de chunking. Divide texto en chunks utilizando un splitter
    recursivo basado en caracteres, pero con conteo de tokens real
    mediante tiktoken."""
    def __init__(self, model_name: str = "gpt-4o-mini") -> None:
        """
        Inicializa el encoder de tokens según el modelo seleccionado.

        Args:
            model_name (str): Modelo usado para definir tokenización.
        """
        self.model_name = model_name
        self.encoder = get_encoder(model_name)

    def _count_tokens(self, text: str) -> int:
        """
        Cuenta el número de tokens en un texto utilizando el encoder. Esto
        permite que el chunking se base en tokens reales en lugar de
        caracteres.

        Args:
            text (str): Texto para contar tokens.

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

        # Para cada página en la lista de páginas del documento...
        for page in pages:
            
            # Se extrae el nombre, el número y el texto de la
            # página del documento
            source = page.get("source")
            page_num = page.get("page")
            text = page.get("text") or ""

            # Si no hay texto en la página, se continúa con la siguiente
            # página
            if not text:
                continue

            # Se fragmenta el texto en chunks utilizando el splitter recursivo
            split_texts = splitter.split_text(text)

            # Para cada índice local y fragmento de texto...
            for index, split_text in enumerate(split_texts):
                
                # Si el chunk tiene menos de 20 tokens, se considera pequeño y
                # se omite para evitar generar chunks irrelevantes
                if self._count_tokens(split_text) < 20:
                    continue
                
                # Se genera un ID único para el chunk utilizando UUID5, basado
                # en el nombre del documento, página del documento y el texto
                # del chunk para asegurar trazabilidad y evitar colisiones
                chunk_id = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_DNS,
                        f"{source}|{page_num}|{index}"
                    )
                )

                # Se añade a la lista un Documento con el texto como contenido
                # y los metadatos de página, archivo fuente y ID del chunk
                # para trazabilidad
                chunks.append(
                    Document(
                        page_content=split_text,
                        metadata={
                            "page": page_num,
                            "source": source,
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