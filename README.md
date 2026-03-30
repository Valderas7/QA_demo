# RAG Document QA API

Sistema de preguntas y respuestas sobre documentos PDF utilizando Retrieval-Augmented Generation (RAG) con FastAPI.

## Descripción

Esta API permite ingestar documentos PDF, procesarlos en chunks, almacenarlos en una base de datos vectorial y realizar consultas en lenguaje natural sobre el contenido. Utiliza modelos de embedding, re-ranking y LLMs para proporcionar respuestas precisas basadas en el contexto de los documentos.

## Características

- **Ingesta de PDFs**: Carga y procesa documentos PDF con detección automática de duplicados
- **Chunking inteligente**: División del texto en fragmentos semánticamente coherentes
- **Embeddings**: Generación de vectores semánticos con modelos de HuggingFace
- **Búsqueda Híbrida**: Combinación de búsqueda vectorial semántica (Qdrant) y léxica (BM25) para máxima precisión
- **Re-ranking**: Mejora de la relevancia de los resultados antes de la generación
- **Generación de respuestas**: LLM local con Ollama para respuestas contextualizadas
- **Timeout de seguridad**: Límite de 5 minutos por consulta para evitar cuelgues
- **API REST**: Interfaz sencilla con FastAPI
- **Logging estructurado**: Seguimiento detallado de operaciones
- **Deduplicación de documentos**: Evita indexar documentos duplicados automáticamente

## Tecnologías

- **FastAPI**: Framework web moderno y rápido
- **LangChain**: Framework para aplicaciones con LLMs
- **Qdrant**: Base de datos vectorial con persistencia
- **BM25**: Búsqueda léxica eficiente (índice invertido)
- **Sentence Transformers**: Modelos de embedding
- **Ollama**: Servidor de LLMs local
- **PDFPlumber**: Extracción de texto de PDFs
- **Uvicorn**: Servidor ASGI

## Instalación

### Requisitos previos

- Python 3.11+
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

Antes de poder realizar consultas, debes ingestar todos los PDFs disponibles en `data/pdfs/`. El sistema indexa automáticamente los documentos en la búsqueda semántica y léxica.

Puedes hacerlo desde la documentación interactiva (`/docs`) o con curl:

```bash
# Ejemplo: ingestar un PDF
curl -X POST "http://localhost:8000/ingest" \
  -F "file=@data/pdfs/documento.pdf"

# Repite para cada PDF en la carpeta
```

**Nota importante:** El sistema detecta automáticamente si un documento ya está indexado y evita duplicados. Cada PDF ingresado amplía el índice vectorial e índice léxico (BM25) existente.

3. **Realizar consultas**

Una vez ingresados los PDFs, ya puedes hacer consultas sobre el contenido desde `/docs` o con curl:

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "query=Tu pregunta aquí"
```

### Endpoints

#### 1. Ingestar PDF

**POST** `/ingest`

Sube un documento PDF para procesarlo e indexarlo en los índices semántico y léxico.

**Parámetros:**
- `file`: Archivo PDF (multipart/form-data)

**Ejemplo con curl:**
```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@documento.pdf"
```

**Respuesta (documento nuevo):**
```json
{
  "pages": 10,
  "chunks": 42,
  "status": "Indexado"
}
```

**Respuesta (documento ya existente):**
```json
{
  "status": "Ya indexado",
  "message": "El documento 'documento.pdf' ya existe en el índice.",
  "pages": 0,
  "chunks": 0
}
```

#### 2. Realizar consulta

**POST** `/query`

Realiza una consulta en lenguaje natural sobre los documentos indexados usando búsqueda híbrida (semántica + léxica).

**Parámetros (form-data):**
- `query`: Consulta en lenguaje natural (requerido)

**Ejemplo con curl:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "query=¿Cuál es el tema principal del documento?"
```

**Respuesta:**
```json
{
  "answer": "El tema principal del documento es...",
  "sources": [
    {
      "text": "Fragmento relevante del documento...",
      "source": "documento.pdf",
      "page": 1
    },
    {
      "text": "Otro fragmento relevante...",
      "source": "documento.pdf",
      "page": 5
    }
  ]
}
```

**Notas:**
- El timeout máximo para generar una respuesta es de 5 minutos
- La búsqueda híbrida combina resultados semánticos (vectoriales) y léxicos para mayor precisión
- Los chunks se reordenan automáticamente usando un modelo de re-ranking antes de generar la respuesta

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
│   ├── constants.py       # Constantes y configuración
│   ├── dependencies.py    # Inyección de dependencias (servicios)
│   ├── exceptions.py      # Excepciones personalizadas
│   ├── logging.py         # Configuración de logging
│   └── prompts.py         # Plantillas de prompts para LLM
├── services/
│   ├── chunking.py        # Servicio de dividir texto en chunks
│   ├── embedding.py       # Servicio de generación de embeddings
│   ├── ingestion.py       # Servicio de ingesta de PDFs
│   ├── llm_service.py     # Servicio de modelo de lenguaje (LLM)
│   ├── reranker.py        # Servicio de re-ranking de resultados
│   └── search/
│       ├── hybrid.py      # Búsqueda híbrida (semántica + léxica)
│       ├── lexical.py     # Búsqueda léxica (BM25)
│       └── semantic.py    # Búsqueda semántica vectorial (Qdrant)
├── data/
│   ├── pdfs/              # Directorio para almacenar PDFs
│   ├── qdrant_db/         # Base de datos vectorial Qdrant (persistencia)
│   └── eval/              # Datos de evaluación
│       ├── eval.jsonl     # Conjunto de evaluación en formato JSONL
│       └── responses/     # Respuestas generadas por el RAG
└── tests/
    └── dataset.py         # Utilidades para trabajar con datasets
```

## Configuración

### Modelos de embedding

Por defecto, el proyecto utiliza modelos de Sentence Transformers. Puedes configurar el modelo en [services/embedding.py](services/embedding.py).

### Modelo LLM

Este proyecto utiliza **gemma2:2b** como modelo de lenguaje a través de Ollama. 

Asegúrate de tener el modelo descargado:
```bash
ollama pull gemma2:2b
```

Si deseas usar otro modelo, configúralo en [services/llm_service.py](services/llm_service.py). Otros modelos compatibles:
- llama2
- mistral
- phi

### Base de datos vectorial

El proyecto utiliza Qdrant como base de datos vectorial. Los datos se persisten en `data/qdrant_db/`. La configuración se puede modificar en los servicios de búsqueda (`services/search/`).


## Logs

Los logs se generan en formato JSON estructurado para facilitar su análisis. La configuración se encuentra en [core/logging.py](core/logging.py).


## Decisiones Técnicas

El sistema implementa un pipeline RAG completo que combina recuperación de información con generación de respuestas:

1. **Ingesta y Procesamiento**
   - **PDFPlumber** para extracción de texto: Elegido por su precisión en la extracción de texto estructurado de PDFs
   - **Chunking estratégico**: División del texto en fragmentos semánticamente coherentes para optimizar la recuperación

2. **Embedding y Búsqueda Vectorial**
   - **Sentence Transformers**: Modelos pre-entrenados de HuggingFace para generar embeddings de alta calidad
   - **Qdrant**: Base de datos vectorial eficiente con búsqueda por similitud, almacenamiento persistente en disco (`data/qdrant_db/`) y bajo overhead de latencia
   - **BM25 (Búsqueda Léxica)**: Índice de recuperación de información basado en frecuencia de términos para búsqueda exacta y por palabra clave
   - **Búsqueda Híbrida**: Combinación inteligente de resultados de búsqueda semántica (Qdrant) y léxica (BM25) para maximizar precisión y recall en la recuperación de documentos

3. **Re-ranking**
   - Capa adicional de refinamiento tras la recuperación inicial, ya que evalúa cada par (query, documento) de forma conjunta, mejorando la relevancia de los documentos seleccionados antes de la generación
   - Reduce el ruido y falsos positivos en el contexto enviado al LLM
   - Se aplica automáticamente con los 3 mejores documentos tras la búsqueda híbrida

4. **Deduplicación de Documentos**
   - El sistema detecta automáticamente si un documento ya ha sido indexado en ambos servicios (semántico y léxico)
   - Evita procesar y almacenar duplicados, ahorrando espacio y tiempo de indexación
   - Mantiene la integridad del índice sin duplicados

5. **Timeout de Seguridad**
   - Límite máximo de 5 minutos (300 segundos) por consulta para evitar que el proceso se cuelgue indefinidamente
   - Devuelve error 504 (Gateway Timeout) si se excede el tiempo límite
   - Essential para mantener la disponibilidad de la API en producción

6. **Modelo de Lenguaje**
   - **Ollama con gemma2:2b**
   - Ventaja: Modelo ligero (2B parámetros) ejecutado localmente sin dependencias externas
   - Desventaja: Menor capacidad comparado con modelos más grandes (7B+)

7. **Framework y API**
   - **LangChain**: Abstracción de alto nivel para pipelines LLM
   - **FastAPI**: Framework moderno con validación automática (Pydantic), documentación OpenAPI y alto rendimiento
   - **Arquitectura modular**: Separación clara entre servicios (ingesta, query, embeddings, etc.)

### Logging Estructurado

- Formato JSON para facilitar análisis y monitorización
- En este formato tiene una mejor integración con sistemas de observabilidad.

### Escalabilidad

- **Actual**: Qdrant con persistencia local, escalable para datasets medianos
- **Búsqueda Híbrida**: Combinación de estrategias semánticas (vectoriales) y léxicas para mayor precisión
- **Modular**: Arquitectura desacoplada que permite reemplazar componentes (embeddings, LLM, retriever) fácilmente
