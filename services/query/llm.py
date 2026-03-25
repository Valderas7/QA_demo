# Librerías
import logging
from typing import List
from langchain_community.llms import ollama
from langchain_core.documents import Document

# Logger del módulo
logger = logging.getLogger(__name__)


class LLMService:
    """
    Servicio de generación de respuestas usando un LLM local (Ollama).

    Este servicio construye un prompt a partir de documentos recuperados
    (chunks) y genera una respuesta basada exclusivamente en ese contexto.
    Incluye citación de fuente y número de página.
    """

    def __init__(self, model_name: str = "mistral") -> None:
        """
        Inicializa el cliente de LLM.

        Args:
            model_name (str): Nombre del modelo disponible en Ollama.
        """
        self.model_name = model_name
        self.llm = ollama(model=model_name)

    def generate(self, query: str, docs: List[Document]) -> str:
        """
        Genera una respuesta a partir de una consulta y documentos relevantes.

        Construye un prompt que incluye contexto recuperado (chunks) y
        fuerza al modelo a responder únicamente con dicha información,
        incluyendo citas de fuente y página.

        Args:
            query (str): Pregunta del usuario en lenguaje natural.
            docs (List[Document]): Lista de documentos recuperados por el
            retriever.
            Cada documento debe contener:
                - page_content (str)
                - metadata (dict con 'source' y 'page')

        Returns:
            str: Respuesta generada por el modelo.
        """
        # Si no hay documentos recuperados se devuelve un string
        if not docs:
            logger.warning("No se recibieron documentos para generar contexto.")
            return "No encontrado en los documentos proporcionados."

        # Se cosntruye el contexto
        context = "\n\n".join([
            f"[Fuente: {d.metadata.get('source')} - pág {d.metadata.get('page')}]\n"
            f"{d.page_content}"
            for d in docs
        ])

        # Prompt controlado (RAG)
        prompt = f"""
Responde a la pregunta usando SOLO el contexto proporcionado.

Contexto:
{context}

Pregunta:
{query}

Instrucciones:
- Responde de forma clara y precisa.
- Cita siempre la fuente y número de página.
- Si la información no está en el contexto, responde: "No encontrado".
"""

        logger.info(
            f"Generando respuesta con {len(docs)} documentos | modelo={self.model_name}"
        )

        try:
            response: str = self.llm.invoke(prompt)

            logger.debug("Respuesta generada correctamente.")
            return response

        except Exception as e:
            logger.error(f"Error generando respuesta: {e}")
            raise