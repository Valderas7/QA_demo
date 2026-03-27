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
            source (str): Nombre del archivo.

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

                    # Se extrae el texto
                    raw_text = page.extract_text()

                    # Si no hay texto, se muestra una advertencia y se
                    # continúa con la siguiente página
                    if not raw_text:
                        logger.warning(
                            f"La página {i + 1} del archivo '{source}' "
                            "no contiene capa de texto (necesario OCR)."
                        )
                        continue
                    
                    # Se limpia el texto eliminando espacios al inicio y
                    # al final
                    raw_text = raw_text.strip()

                    # Se divide el texto y se vuelve a unir para eliminar
                    # palabras raras
                    text = " ".join(raw_text.split())
                    
                    # Se añade a la lista un diccionario con el texto,
                    # el número de página y la fuente
                    pages.append({
                        "text": text,
                        "page": i + 1,
                        "source": source
                    })

            # Mensaje de logging con el número de páginas procesadas
            logger.info(
                f"{len(pages)} páginas procesadas del archivo "
                f"'{source}'."
            )

            # Se devuelve la lista de diccionario de páginas
            return pages
        
        # Excepción
        except Exception:
            logger.exception(f"Error procesando el PDF '{source}'.")
            raise