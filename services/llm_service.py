# Librerías
import logging
from core.prompts import RAGPromptTemplate
from typing import List
from langchain_ollama import OllamaLLM
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

    def __init__(self, model_name: str = "gemma2:2b") -> None:
        """
        Inicializa el cliente de LLM.

        Args:
            model_name (str): Nombre del modelo disponible en Ollama.
        """
        self.llm = OllamaLLM(model=model_name, temperature=0)

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

        # Se construye el contexto uniendo la fuente (el PDF), la página donde
        # aparece y el texto para cada uno de los documentos de Langchain
        context = "\n\n".join([
            f"[Fuente: {doc.metadata.get('source')} - pág {doc.metadata.get('page')}]\n"
            f"{doc.page_content}"
            for doc in docs
        ])

        # Se genera el prompt a partir del contexto y la consulta
        prompt = RAGPromptTemplate.generate_prompt(context, query)

        # Se intenta invocar al modelo
        try:
            response: str = self.llm.invoke(prompt)
            logger.info("Respuesta generada correctamente.")
            return response

        # Excepción
        except Exception as e:
            logger.error(f"Error generando respuesta: {e}.")
            raise