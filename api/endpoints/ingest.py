# Librerías
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from services.ingestion.ingest import IngestService
from services.ingestion.chunking import ChunkService
from services.ingestion.embedding import EmbeddingService
from services.ingestion.vector_store import VectorStoreService

# Se obtiene el logger para este módulo
logger = logging.getLogger(__name__)

# Se crea un enrutador
router = APIRouter()

# Inicia servicios
ingestor = IngestService()
chunker = ChunkService()
embedder = EmbeddingService()
vectorstore = VectorStoreService(embedder)

# Endpoint para ingestar PDF y crear índice vectorial con los chunks del PDF
@router.post("/ingest", tags=["Ingesta"])
async def ingest_pdf(file: UploadFile = File(...)):

    # Se intenta...
    try:

        # Validación básica
        if not file.filename.endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail="Solo se permiten archivos PDF"
            )

        # Se lee el PDF
        content = await file.read()

        # Se llama al método para ingestar PDFs
        pages = ingestor.load_pdf(
            file=content,
            source=file.filename
        )

        # Chunkea el texto del PDF
        chunks = chunker.chunk_text(pages)

        # Se separan los textos y metadatos de los chunks
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [
            {
                "page": chunk["page"],
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"]
            }
            for chunk in chunks
        ]

        # Se guarda en la BBDD vectorial los embeddings a partir de los
        # chunks obtenidos
        vectorstore.add(texts, metadatas)

        # Devuelve un diccionario con los chunks indexados
        return {
            "message": "Documento ingerido correctamente",
            "file": file.filename,
            "chunks": len(chunks)
        }
    
    # Excepción
    except Exception as e:
        logger.error(f"Error ingestando el documento: {e}.")
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando PDF: {str(e)}"
        )



