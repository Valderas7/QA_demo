# Librerías
import logging
import uvicorn
from api.endpoints import ingest, query
from core.logging import setup_logging
from fastapi import FastAPI

# Configuración del logging
setup_logging()

# Se obtiene el logger para este módulo
logger = logging.getLogger(__name__)

# Inicia la APP
app = FastAPI(
    title="RAG Document QA API",
    summary=(
        "API para ingestar PDFs y responder preguntas basadas en su "
        "contenido usando RAG."
    ),
    description=(
        "Esta API permite ingestar documentos PDF, procesarlos para extraer "
        "su contenido, dividirlo en chunks y almacenarlo en una base de "
        "datos vectorial. Luego, permite hacer consultas en lenguaje natural "
        "que se responden utilizando un modelo de lenguaje que sintetiza la "
        "información de los chunks más relevantes recuperados de la base de "
        "datos vectorial."
    ),
    version="0.5.1"
)

# Se incluyen los enrutadores en la APP principal
app.include_router(ingest.router)
app.include_router(query.router)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)