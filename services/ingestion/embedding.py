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
            model_name,
            encode_kwargs={"normalize_embeddings": True}
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Genera embeddings para una lista de textos.

        Args:
            texts (list[str]): Lista de textos a convertir en embeddings.

        Returns:
            list[list[float]]: Lista de vectores de embeddings, donde cada
            vector es una lista de floats que representa el texto en un
            espacio semántico denso.
        """
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """
        Genera embedding para una consulta de texto.

        Args:
            text (str): Texto de la consulta a convertir en embedding.

        Returns:
            list[float]: Vector de embedding que representa la consulta en
            el mismo espacio semántico que los documentos, facilitando la
            comparación y búsqueda semántica.
        """
        return self.embeddings.embed_query(text)