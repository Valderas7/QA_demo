# Librerías
import json
import requests
from typing import List, Dict
import re

# Endpoint para consultas
API_URL = "http://127.0.0.1:8000/query"


def load_eval() -> List[Dict]:
    """
    Carga los datos de evaluación desde un archivo JSONL.

    Cada línea del archivo debe ser un JSON con al menos los campos:
    - "question": la pregunta a evaluar.
    - "expected_answer": la respuesta esperada.

    Returns:
        List[Dict]: Lista de diccionarios con las preguntas y respuestas esperadas.
    """
    with open("data/eval/eval.jsonl", "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def run_query(question: str) -> Dict:
    """
    Realiza una consulta al endpoint definido y devuelve la respuesta del
    modelo.

    Args:
        question (str): La pregunta a enviar al endpoint.

    Returns:
        Dict: Diccionario con la respuesta del modelo y los documentos
        retornados.
        Ejemplo: {"response": "...", "documents": [{"source": "...", "page": 1}, ...]}
    """
    resp = requests.get(API_URL, params={"q": question, "top_k": 5, "rerank_top_k": 3})
    resp.raise_for_status()
    return resp.json()


def has_valid_citations(response: str, returned_docs: List[Dict]) -> bool:
    """
    Verifica si la respuesta incluye citas válidas a los documentos retornados.

    Se consideran citas válidas patrones como: [Fuente: xxx.pdf - pág 5].

    Args:
        response (str): Texto de la respuesta generada por el modelo.
        returned_docs (List[Dict]): Lista de documentos devueltos por el
        modelo, con campos "source" y "page".

    Returns:
        bool: True si al menos una cita coincide con los documentos
        devueltos, False en caso contrario.
    """
    citations = re.findall(r"Fuente:\s*([^\s-]+?)\s*-\s*pág\s*(\d+)", response)
    if not citations:
        return False
    
    returned_set = {(doc["source"], str(doc["page"])) for doc in returned_docs}
    for src, page in citations:
        if (src.strip(), page.strip()) in returned_set:
            return True
    return False

def evaluate():
    """
    Ejecuta la evaluación completa del modelo sobre el conjunto de preguntas.

    Métricas calculadas:
    - Citation Accuracy: porcentaje de respuestas con citas válidas.

    También imprime los resultados y guarda un archivo JSON con información
    detallada de cada pregunta evaluada.
    """
    data = load_eval()
    total = len(data)
    correct_citations = 0
    responses = []

    print(f"Evaluando {total} preguntas...\n")

    for item in data:
        result = run_query(item["question"])
        response_text = result["response"]
        returned_docs = result["documents"]

        # Métrica 1: % de respuestas con citas válidas
        valid_cit = has_valid_citations(response_text, returned_docs)
        if valid_cit:
            correct_citations += 1

        # Guardamos para después (puedes añadir LLM-as-judge aquí si quieres)
        responses.append({
            "question": item["question"],
            "generated": response_text,
            "expected": item["expected_answer"],
            "valid_citation": valid_cit,
            "docs_returned": len(returned_docs)
        })

        print(f"✅ {item['question'][:60]}... → cita válida: {valid_cit}")

    # Métricas finales
    citation_accuracy = (correct_citations / total) * 100
    print("\n" + "="*50)
    print("MÉTRICAS FINALES")
    print(f"• Citation Accuracy: {citation_accuracy:.1f}%")
    print(f"• Respuestas evaluadas: {total}")
    print("="*50)

    # Opcional: guardar resultados
    with open("data/eval/results.json", "w", encoding="utf-8") as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    evaluate()