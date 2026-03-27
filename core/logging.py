# Librerías
import logging
from pythonjsonlogger import jsonlogger


# Función para configurar el logging en formato JSON
def setup_logging():

    # Crear un manejador de logs que envíe los logs a la consola (stdout)
    log_handler = logging.StreamHandler()

    # Se crea el formateador JSON para los logs mostrando correctamente
    # los caracteres especiales y con el formato de fecha deseado
    formatter = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(name)s %(message)s',
                                         json_ensure_ascii=False,
                                         datefmt='%d-%m-%Y %H:%M:%S')

    # Asigna dicho formateador al manejador de logs
    log_handler.setFormatter(formatter)

    # Configura el logger raíz para usar el manejador creado
    logger = logging.getLogger()
    logger.addHandler(log_handler)
    logger.setLevel(logging.INFO)

    # Se desactiva el logging de algunas librerías utilizadas para evitar
    # logs excesivos
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("faiss").setLevel(logging.WARNING)