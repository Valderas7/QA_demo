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
    def __init__(self, model_name = "BAAI/bge-base-en-v1.5") -> None:
        """
        Inicializa el modelo de embeddings.

        Carga un modelo preentrenado de SentenceTransformers
        optimizado para tareas de retrieval semántico.

        Args:
            model_name (str): Nombre del modelo de HuggingFace a usar.
            Por defecto se usa "BAAI/bge-base-en-v1.5", un modelo
            eficiente y preciso para embeddings en inglés.
        """
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True}
        )