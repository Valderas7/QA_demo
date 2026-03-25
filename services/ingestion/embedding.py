# Librerías
from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingService:
    """
    Servicio de embeddings para pipeline RAG.

    Este servicio encapsula un modelo de embeddings basado en
    HuggingFace para generar representaciones vectoriales densas
    de textos. Es utilizado en combinación con FAISS u otros
    vector stores para búsqueda semántica.
    """
    def __init__(self) -> None:
        """
        Inicializa el modelo de embeddings.

        Carga un modelo preentrenado de SentenceTransformers
        optimizado para tareas de retrieval semántico.
        """
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5"
        )

    def get(self) -> HuggingFaceEmbeddings:
        """
        Devuelve la instancia del modelo de embeddings.

        Returns:
            HuggingFaceEmbeddings: Modelo listo para ser usado
            en FAISS o cualquier vector database compatible.
        """
        return self.embeddings