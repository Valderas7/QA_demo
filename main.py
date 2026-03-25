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
app = FastAPI(title="RAG Document QA API")

# Se incluyen los enrutadores en la APP principal
app.include_router(ingest.router)
app.include_router(query.router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)