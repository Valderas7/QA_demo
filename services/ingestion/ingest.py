# Librerías
import io
import logging
import pdfplumber
from typing import Any, List, Dict

# Se obtiene el logger para este módulo
logger = logging.getLogger(__name__)


class IngestService:
    """Clase para ingestar PDFs"""

    def load_pdf(self, file: bytes, source: str) -> List[Dict[str, Any]]:
        """
        Extrae el contenido textual de un archivo PDF en memoria y lo
        estructura por páginas.

        Args:
            file (bytes): Contenido binario del archivo PDF.
            source (str): Nombre o identificador del documento (ej: nombre
            del archivo).

        Returns:
            list[dict]: Lista de páginas con la estructura:
                {
                    "text": str,   # texto limpio de la página
                    "page": int,   # número de página
                    "source": str  # origen del documento
                }
        """
        # Se intenta...
        try:

            # Lista vacía
            pages = []

            # Se abre el archivo PDF con pdfplumber
            with pdfplumber.open(io.BytesIO(file)) as pdf:

                # Para cada página...
                for i, page in enumerate(pdf.pages):

                    # Se extrae el texto o si no, un string vacío
                    text = page.extract_text() or ""

                    # Se divide el texto y se vuelve a unir para eliminar
                    # palabras raras
                    text = " ".join(text.split())
                    
                    # Se añade a la lista un diccionario con el texto,
                    # el número de página y la fuente
                    pages.append({
                        "text": text,
                        "page": i + 1,
                        "source": source
                    })

            # Se devuelve la lista de diccionario de páginas
            logger.info(
                f"{len(pages)} páginas procesadas del archivo "
                f"'{source}'.")
            return pages
        
        # Excepción
        except Exception as e:
            logger.error(f"Error procesando PDF {source}: {e}")
            raise