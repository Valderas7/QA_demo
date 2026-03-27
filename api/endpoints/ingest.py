# Librerías
import logging
from core.dependencies import get_services, Services
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException

# Se obtiene el logger para este módulo
logger = logging.getLogger(__name__)

# Se crea un enrutador
router = APIRouter()


# Endpoint para ingestar PDF y crear índice vectorial con los chunks del PDF
@router.post("/ingest", tags=["Ingesta"])
async def ingest_pdf(
    file: UploadFile = File(...),
    services: Services = Depends(get_services)
):
    """
    Endpoint para ingestar un PDF y crear un índice vectorial con los chunks
    del PDF. El endpoint recibe un archivo PDF, lo procesa para extraer su
    contenido, lo divide en chunks y luego guarda esos chunks en una base de
    datos vectorial para su posterior consulta.
    """
    # Se intenta...
    try:

        # Se valida que el archivo sea un PDF
        if not file.filename.endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail="Solo se permiten archivos PDF"
            )
        
        # Se lee el contenido del archivo PDF
        content = await file.read()

        # Se llama al método para ingestar PDFs, obteniendo una lista de
        # páginas con su texto limpio y metadatos de página y fuente
        pages = services.ingest.load_pdf(content, file.filename)

        # Se fragmenta el texto del PDF en chunks utilizando el servicio de
        # chunking, obteniendo una lista de Documentos de Langchain
        chunks = services.chunk.chunk_text(pages)

        # Se guarda en la base de datos vectorial los embeddings a partir de
        # los chunks obtenidos utilizando el servicio de vector store
        services.vectorstore.add(chunks)

        # Devuelve un mensaje de éxito con el número de páginas procesadas, el
        # número de chunks generados y el estado de la ingesta
        return {
            "pages": len(pages),
            "chunks": len(chunks),
            "status": "indexado correctamente"
        }

    # Excepción
    except Exception as e:
        logger.exception("Error ingestando el documento.")
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando PDF: {str(e)}"
        )



