# RAG Document QA API

Sistema de preguntas y respuestas sobre documentos PDF utilizando Retrieval-Augmented Generation (RAG) con FastAPI.

## Descripción

Esta API permite ingestar documentos PDF, procesarlos en chunks, almacenarlos en una base de datos vectorial y realizar consultas en lenguaje natural sobre el contenido. Utiliza modelos de embedding, re-ranking y LLMs para proporcionar respuestas precisas basadas en el contexto de los documentos.

## Características

- **Ingesta de PDFs**: Carga y procesa documentos PDF
- **Chunking inteligente**: División del texto en fragmentos manejables
- **Embeddings**: Generación de vectores semánticos con modelos de HuggingFace
- **Búsqueda vectorial**: Recuperación eficiente con FAISS
- **Re-ranking**: Mejora de la relevancia de los resultados
- **Generación de respuestas**: LLM local con Ollama para respuestas contextualizadas
- **API REST**: Interfaz sencilla con FastAPI
- **Logging estructurado**: Seguimiento detallado de operaciones

## Tecnologías

- **FastAPI**: Framework web moderno y rápido
- **LangChain**: Framework para aplicaciones con LLMs
- **FAISS**: Búsqueda de similitud vectorial eficiente
- **Sentence Transformers**: Modelos de embedding
- **Ollama**: Servidor de LLMs local
- **PDFPlumber**: Extracción de texto de PDFs
- **Uvicorn**: Servidor ASGI

## Instalación

### Requisitos previos

- Python 3.8+
- [Ollama](https://ollama.ai/) instalado y ejecutándose

### Pasos de instalación

1. **Clonar el repositorio**
```bash
git clone <repository-url>
cd QA_demo
```

2. **Crear entorno virtual**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Configurar Ollama y descargar modelo en local**
```bash
ollama pull gemma2:2b
```

## Uso

### Documentación interactiva

Accede a la documentación Swagger en: `http://localhost:8000/docs`

### Flujo de trabajo completo

1. **Iniciar el servidor**
```bash
python main.py
```

La API estará disponible en `http://localhost:8000`

2. **Ingestar PDFs de la carpeta data/pdfs**

Antes de poder realizar consultas, debes ingestar todos los PDFs disponibles en `data/pdfs/`. Esto construirá y ampliará el índice vectorial.

Puedes hacerlo desde la documentación interactiva (`/docs`) o con curl:

```bash
# Ejemplo: ingestar cada PDF de la carpeta
curl -X POST "http://localhost:8000/ingest" \
  -F "file=@data/pdfs/documento1.pdf"

curl -X POST "http://localhost:8000/ingest" \
  -F "file=@data/pdfs/documento2.pdf"

# Repite para cada PDF en la carpeta
```

**Nota importante:** Cada PDF ingresado amplía el índice vectorial existente. Ingesta todos los documentos relevantes antes de comenzar a hacer consultas.

3. **Realizar consultas**

Una vez ingresados los PDFs, ya puedes hacer consultas sobre el contenido desde `/docs` o con curl:

```bash
curl -X GET "http://localhost:8000/query?q=Tu pregunta aquí"
```

### Endpoints

#### 1. Ingestar PDF

**POST** `/ingest`

Sube un documento PDF para procesarlo e indexarlo.

**Parámetros:**
- `file`: Archivo PDF (multipart/form-data)

**Ejemplo con curl:**
```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@documento.pdf"
```

**Respuesta:**
```json
{
  "message": "Documento ingerido correctamente",
  "file": "documento.pdf",
  "chunks": 42
}
```

#### 2. Realizar consulta

**GET** `/query`

Realiza una consulta en lenguaje natural sobre los documentos indexados.

**Parámetros:**
- `q`: Consulta en lenguaje natural (requerido)
- `top_k`: Número de documentos a recuperar antes del reranking (default: 5)
- `rerank_top_k`: Número de documentos finales tras reranking (default: 3)

**Ejemplo con curl:**
```bash
curl -X GET "http://localhost:8000/query?q=¿Cuál es el tema principal del documento?&top_k=5&rerank_top_k=3"
```

**Respuesta:**
```json
{
  "query": "¿Cuál es el tema principal del documento?",
  "response": "El tema principal del documento es...",
  "documents": [
    {
      "text": "Fragmento relevante del documento...",
      "source": "documento.pdf",
      "page": 1,
      "chunk_id": "doc_0_chunk_0"
    }
  ]
}
```

## Estructura del proyecto

```
QA_demo/
├── main.py                # Punto de entrada de la aplicación
├── requirements.txt       # Dependencias del proyecto
├── README.md              # Este archivo
├── api/
│   └── endpoints/
│       ├── ingest.py      # Endpoint para ingestar PDFs
│       └── query.py       # Endpoint para consultas
├── core/
│   ├── dependencies.py    # Dependencias compartidas (vectorstore)
│   └── logging.py         # Configuración de logging
├── services/
│   ├── ingestion/
│   │   ├── chunking.py    # Servicio de chunking de texto
│   │   ├── embedding.py   # Servicio de embeddings
│   │   ├── ingest.py      # Servicio de ingesta de PDFs
│   │   └── vector_store.py # Servicio de base de datos vectorial
│   └── query/
│       ├── llm_service.py  # Servicio de LLM
│       ├── rag_retriever.py # Recuperador RAG
│       └── reranker.py     # Servicio de re-ranking
├── data/
│   ├── pdfs/              # Directorio de PDFs públicos
│   └── eval/              # Datos de evaluación con su
│       ├── eval.jsonl     # 
│       └── responses/     # Respuestas obtenidas con el RAG
└── tests/
    └── dataset.py         # Script para descargar los PDFs públicos
```

## Configuración

### Modelos de embedding

Por defecto, el proyecto utiliza modelos de Sentence Transformers. Puedes configurar el modelo en `services/ingestion/embedding.py`.

### Modelo LLM

Este proyecto utiliza **gemma2:2b** como modelo de lenguaje a través de Ollama. 

Asegúrate de tener el modelo descargado:
```bash
ollama pull gemma2:2b
```

Si deseas usar otro modelo, configúralo en `services/query/llm_service.py`. Otros modelos compatibles:
- llama2
- mistral
- phi

### Base de datos vectorial

FAISS se inicializa en memoria. Para persistencia, modifica `services/ingestion/vector_store.py`.


## Logs

Los logs se generan en formato JSON estructurado para facilitar su análisis. La configuración se encuentra en `core/logging.py`.
