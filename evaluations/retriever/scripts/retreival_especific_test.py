import json
import sys
import os
from tqdm import tqdm
import pandas as pd
from typing import List, Dict

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.insert(0, project_root)

try:
    from src.lg_nodes import retriever_tool
    print("Import exitoso")
except ImportError as e:
    print(f"Error en import: {e}")
    print("Asegurate de ejecutar desde el directorio correcto")
    sys.exit(1)

def evaluate_retrieval_precision(eval_dataset: List[Dict], k_values: List[int]) -> pd.DataFrame:
    """
    Evalúa la precisión del retrieval - si trae los documentos correctos y en qué posición.
    
    Métricas:
    - Recall@k: ¿Qué porcentaje de documentos esperados están en los top-k?
    - Precision@k: ¿Qué porcentaje de los top-k documentos son relevantes?
    - MRR (Mean Reciprocal Rank): ¿En qué posición aparece el primer documento relevante?
    - Hit Rate@k: ¿En cuántas consultas se encontró al menos 1 documento relevante?
    """
    
    print("Evaluando precisión del retrieval...")
    
    all_results = []
    
    for k in k_values:
        print(f"\n--- Evaluando con k={k} ---")
        
        recall_scores = []
        precision_scores = []
        reciprocal_ranks = []
        hits = []
        position_analysis = []
        
        for i, item in enumerate(tqdm(eval_dataset, desc=f"k={k}")):
            question = item["question"]
            expected_sources = item["source_chunks"]
            
            try:
                retrieved_docs = retriever_tool(question)
                
                if not retrieved_docs:
                    print(f"No se recuperaron documentos para: {question[:50]}...")
                    recall_scores.append(0.0)
                    precision_scores.append(0.0)
                    reciprocal_ranks.append(0.0)
                    hits.append(0)
                    continue
                    
            except Exception as e:
                print(f"Error en retrieval para '{question[:50]}...': {e}")
                recall_scores.append(0.0)
                precision_scores.append(0.0)
                reciprocal_ranks.append(0.0)
                hits.append(0)
                continue
            
            top_k_docs = retrieved_docs[:k]
            
            relevant_docs_found = set()  # Documentos que son relevantes
            expected_chunks_found = set()  # Chunks esperados que se encontraron
            match_positions = []
            
            for pos, doc in enumerate(top_k_docs):
                doc_is_relevant = False
                for expected_chunk in expected_sources:
                    if _is_relevant_match(doc, expected_chunk):
                        expected_chunks_found.add(expected_chunk)
                        if not doc_is_relevant:  
                            relevant_docs_found.add(pos)
                            match_positions.append(pos + 1)
                            doc_is_relevant = True

            
            num_expected_chunks = len(expected_sources)
            num_expected_chunks_found = len(expected_chunks_found)
            num_relevant_docs_found = len(relevant_docs_found)
            
            # Recall@k: porcentaje de chunks esperados que se encontraron
            recall = num_expected_chunks_found / num_expected_chunks if num_expected_chunks > 0 else 0.0
            
            # Precision@k: porcentaje de documentos top-k que son relevantes
            precision = num_relevant_docs_found / k if k > 0 else 0.0
            
            # Reciprocal Rank: 1/posición del primer match (0 si no hay matches)
            rr = 1.0 / match_positions[0] if match_positions else 0.0
            
            # Hit: 1 si encontró al menos un documento relevante
            hit = 1 if num_relevant_docs_found > 0 else 0
            
            recall_scores.append(recall)
            precision_scores.append(precision)
            reciprocal_ranks.append(rr)
            hits.append(hit)
            
            position_analysis.append({
                'question_id': i,
                'question': question[:100] + "..." if len(question) > 100 else question,
                'expected_chunks': num_expected_chunks,
                'found_chunks': num_expected_chunks_found,
                'relevant_docs': num_relevant_docs_found,
                'match_positions': match_positions,
                'recall': recall,
                'precision': precision,
                'reciprocal_rank': rr
            })
        
        # Calcular métricas promedio para este k
        avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
        avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
        avg_mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
        hit_rate = sum(hits) / len(hits) if hits else 0.0
        
        print(f" Resultados k={k}:")
        print(f"  Recall@{k}: {avg_recall:.4f}")
        print(f"  Precision@{k}: {avg_precision:.4f}")
        print(f"  MRR@{k}: {avg_mrr:.4f}")
        print(f"  Hit Rate@{k}: {hit_rate:.4f}")
        
        # Guardar resultados
        all_results.append({
            'k': k,
            'recall_at_k': avg_recall,
            'precision_at_k': avg_precision,
            'mrr_at_k': avg_mrr,
            'hit_rate_at_k': hit_rate,
            'total_queries': len(eval_dataset),
            'successful_queries': len([r for r in recall_scores if r > 0])
        })
        
        if k == max(k_values):  # Solo para el k más alto
            df_detailed = pd.DataFrame(position_analysis)
            output_detailed = os.path.join(current_dir, "resultados", f"retrieval_detailed_analysis_k{k}.csv")
            os.makedirs(os.path.dirname(output_detailed), exist_ok=True)
            df_detailed.to_csv(output_detailed, index=False)
            print(f"Análisis detallado guardado en: {output_detailed}")
    
    return pd.DataFrame(all_results)

def _is_relevant_match(retrieved_doc: str, expected_chunk: str, threshold: float = 0.7) -> bool:
    """
    Determina si un documento recuperado coincide con el chunk esperado.
    Usa coincidencia de texto flexible.
    """
    retrieved_clean = _normalize_text(retrieved_doc)
    expected_clean = _normalize_text(expected_chunk)
    
    if expected_clean in retrieved_clean or retrieved_clean in expected_clean:
        return True
    
    expected_words = set(expected_clean.split())
    retrieved_words = set(retrieved_clean.split())
    
    stop_words = {'de', 'la', 'el', 'en', 'y', 'a', 'que', 'es', 'se', 'del', 'las', 'los', 'con', 'por', 'para', 'un', 'una', 'como', 'su', 'al', 'lo', 'le', 'te', 'me', 'nos'}
    expected_words = expected_words - stop_words
    retrieved_words = retrieved_words - stop_words
    
    if len(expected_words) == 0:
        return False
    
    overlap = len(expected_words.intersection(retrieved_words))
    overlap_ratio = overlap / len(expected_words)
    
    return overlap_ratio >= threshold

def _normalize_text(text: str) -> str:
    """Normaliza texto para comparación."""
    import re
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def analyze_position_distribution(eval_dataset: List[Dict], k: int = 10) -> Dict:
    """
    Analiza en qué posiciones aparecen típicamente los documentos relevantes.
    """
    print(f"\nAnalizando distribución de posiciones (k={k})...")
    
    position_counts = {i: 0 for i in range(1, k+1)}
    not_found_count = 0
    first_positions = []
    
    for item in tqdm(eval_dataset, desc="Analizando posiciones"):
        question = item["question"]
        expected_sources = item["source_chunks"]
        
        try:
            retrieved_docs = retriever_tool(question)
            if not retrieved_docs:
                not_found_count += 1
                continue
                
            found_positions = []
            for pos, doc in enumerate(retrieved_docs[:k]):
                for expected_chunk in expected_sources:
                    if _is_relevant_match(doc, expected_chunk):
                        position = pos + 1
                        if position not in found_positions: 
                            found_positions.append(position)
                            position_counts[position] += 1
            
            if found_positions:
                first_positions.append(min(found_positions))
            else:
                not_found_count += 1
                
        except Exception as e:
            print(f"Error: {e}")
            not_found_count += 1
    
    return {
        'position_distribution': position_counts,
        'not_found_count': not_found_count,
        'first_positions': first_positions,
        'avg_first_position': sum(first_positions) / len(first_positions) if first_positions else None
    }

def generate_retrieval_report(results_df: pd.DataFrame, position_analysis: Dict, output_path: str):
    """Genera un reporte completo del rendimiento del retrieval."""
    
    lines = []
    lines.append("REPORTE DE EVALUACIÓN DEL RAG - SOLO RETRIEVAL")
    lines.append("=" * 60)
    lines.append(f"Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Métricas principales por k
    lines.append("=== MÉTRICAS POR VALOR DE K ===")
    lines.append(f"{'k':<3} {'Recall':<8} {'Precision':<10} {'MRR':<8} {'Hit Rate':<10}")
    lines.append("-" * 45)
    
    for _, row in results_df.iterrows():
        lines.append(f"{row['k']:<3} {row['recall_at_k']:<8.3f} {row['precision_at_k']:<10.3f} {row['mrr_at_k']:<8.3f} {row['hit_rate_at_k']:<10.3f}")
    
    # Análisis de posiciones
    if position_analysis:
        lines.append(f"\n=== ANÁLISIS DE POSICIONES ===")
        lines.append(f"Distribución de documentos relevantes por posición:")
        
        for pos, count in position_analysis['position_distribution'].items():
            percentage = count / sum(position_analysis['position_distribution'].values()) * 100 if sum(position_analysis['position_distribution'].values()) > 0 else 0
            lines.append(f"  Posición {pos}: {count} documentos ({percentage:.1f}%)")
        
        if position_analysis['avg_first_position']:
            lines.append(f"\nPosición promedio del primer documento relevante: {position_analysis['avg_first_position']:.2f}")
        
        lines.append(f"Consultas sin documentos relevantes: {position_analysis['not_found_count']}")
    
    # Recomendaciones
    lines.append(f"\n=== RECOMENDACIONES ===")
    
    best_k = results_df.loc[results_df['recall_at_k'].idxmax(), 'k']
    best_recall = results_df['recall_at_k'].max()
    
    if best_recall < 0.5:
        lines.append("RECALL BAJO: El sistema recupera menos del 50% de documentos relevantes")
        lines.append("   - Considera ajustar la estrategia de embeddings")
        lines.append("   - Revisa el preprocesamiento de documentos")
    elif best_recall < 0.7:
        lines.append("RECALL MODERADO: Hay margen de mejora en la recuperación")
        lines.append("   - Considera aumentar el valor de k")
        lines.append("   - Evalúa técnicas de re-ranking")
    else:
        lines.append("RECALL BUENO: El sistema recupera la mayoría de documentos relevantes")
    
    lines.append(f"\n Valor de k recomendado: {best_k} (Recall: {best_recall:.3f})")
    
    # Guardar reporte
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"Reporte guardado en: {output_path}")

def main():
    """Función principal para ejecutar la evaluación del RAG."""
    
    print("EVALUACIÓN DEL RAG - SOLO RETRIEVAL")
    print("=" * 50)
    
    resultados_dir = os.path.join(project_root, "evaluations", "retriever", "results")
    os.makedirs(resultados_dir, exist_ok=True)
    
    dataset_path = os.path.join(current_dir, "eval_dataset_rag.json")
    if not os.path.exists(dataset_path):
        possible_paths = [
            os.path.join(current_dir, "..", "eval_dataset_rag.json"),
            os.path.join(current_dir, "data", "eval_dataset_rag.json"),
            os.path.join(project_root, "evaluations", "retriever", "data", "eval_dataset_rag.json")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                dataset_path = path
                break
        else:
            print("No se encontró eval_dataset_rag.json")
            print("Ubicaciones buscadas:")
            for path in [dataset_path] + possible_paths:
                print(f"  - {path}")
            return
    
    print(f"Dataset encontrado: {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    
    print(f"Cargadas {len(eval_data)} preguntas para evaluación")
    
    k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    
    print("\nIniciando evaluación de precisión...")
    results_df = evaluate_retrieval_precision(eval_data, k_values)
    
    results_path = os.path.join(resultados_dir, "rag_retrieval_evaluation.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Resultados principales guardados en: {results_path}")
    
    position_analysis = analyze_position_distribution(eval_data, k=10)
    
    report_path = os.path.join(resultados_dir, "rag_retrieval_report.txt")
    generate_retrieval_report(results_df, position_analysis, report_path)
    
    print("\n" + "=" * 60)
    print("RESUMEN FINAL")
    print("=" * 60)
    print(results_df.to_string(index=False))
    
    if position_analysis['avg_first_position']:
        print(f"\n Posición promedio del primer documento relevante: {position_analysis['avg_first_position']:.2f}")
    
    print("\n Evaluación completa terminada!")

if __name__ == "__main__":
    main()