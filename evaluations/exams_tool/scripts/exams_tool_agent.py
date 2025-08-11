import os
import sys
import pandas as pd
from typing import List, Dict
from tqdm import tqdm
import evaluate
import re

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.insert(0, project_root)

from src.lg_agent import stream_graph_updates
from langchain.chat_models import init_chat_model
from langchain.evaluation import load_evaluator
from src.lg_nodes import MODEL_NAME, AZURE_DEPLOYMENT_NAME

bertscore = evaluate.load("bertscore")

def clean(text: str) -> str:
    return re.sub(r'\[\d+\]', '', text).strip()

def run_agent_test(test_cases: List[Dict[str, str]], output_csv: str):
    """Evalúa el agente con exams tool usando métricas del RAG."""
    
    llm_for_judge = init_chat_model(
        MODEL_NAME,
        azure_deployment=AZURE_DEPLOYMENT_NAME,
        model_provider="azure_openai",
        verbose=False,
    )
    llm_judge_eval = load_evaluator("cot_qa", llm=llm_for_judge, chain_kwargs={"verbose": False})
    
    results = []

    for case in tqdm(test_cases, desc="Evaluando agente completo con exam queries"):
        query = case["query"]
        expected = clean(case["expected_answer"])
        
        test_user_id = f"test_user_{hash(query) % 10000}"

        try:
            agent_response = stream_graph_updates(
                user_input=query,
                user_id=test_user_id
            )
            output_clean = clean(str(agent_response))
            
        except Exception as e:
            print(f"Error con query '{query}': {e}")
            output_clean = ""
        
        # LLM Judge
        try:
            judge_res = llm_judge_eval.evaluate_strings(
                prediction=output_clean,
                input=query,
                reference=expected,
            )
            judge_score = judge_res.get("score", 0.0)
        except Exception as e:
            print(f"Error en LLM Judge: {e}")
            judge_score = 0.0

        # BERTScore
        try:
            bs = bertscore.compute(
                predictions=[output_clean],
                references=[expected],
                lang="es"
            )
            bert_precision = bs["precision"][0]
            bert_recall = bs["recall"][0]
            bert_f1 = bs["f1"][0]
        except Exception as e:
            print(f"Error en BERTScore: {e}")
            bert_precision = bert_recall = bert_f1 = 0.0

        results.append({
            "query": query,
            "expected_answer": expected,
            "agent_response": output_clean,
            "LLMJudge_score": judge_score,     
            "BERT_prec": bert_precision,       
            "BERT_rec": bert_recall,          
            "BERT_f1": bert_f1,                
            "test_type": case.get("test_type", "basic")
        })

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    print(f"\nResultados guardados en: {output_csv}")
    print("=== MÉTRICAS PROMEDIO ===")
    print(f"LLM Judge: {df['LLMJudge_score'].mean():.4f}")
    print(f"BERT F1: {df['BERT_f1'].mean():.4f}")
    print(f"BERT Precision: {df['BERT_prec'].mean():.4f}")
    print(f"BERT Recall: {df['BERT_rec'].mean():.4f}")

    print("\n=== ANÁLISIS POR TIPO DE TEST ===")
    for test_type in df['test_type'].unique():
        subset = df[df['test_type'] == test_type]
        print(f"{test_type}: LLM_Judge={subset['LLMJudge_score'].mean():.3f}, "
              f"BERT_F1={subset['BERT_f1'].mean():.3f}")
    
    print("\n=== CASOS CON BAJA PUNTUACIÓN ===")
    low_performance = df[df['LLMJudge_score'] < 0.5]
    for _, row in low_performance.iterrows():
        print(f"Query: {row['query']}")
        print(f"Esperado: {row['expected_answer'][:100]}...")
        print(f"Obtenido: {row['agent_response'][:100]}...")
        print(f"LLM Judge: {row['LLMJudge_score']:.3f}")
        print("-" * 50)
    
    return df

if __name__ == "__main__":
    output_file = os.path.join(project_root, "evaluations", "exams_tool", "results", f"exam_tool_with_agent_test.csv")

    TEST_CASES = [
        #tests de reconocimiento de materias
        {
            "query": "¿Cuándo es el examen de Fundamentos de Administración?",
            "expected_answer": "El examen de Fundamentos de Administración (A016) se llevó a cabo el 30 de abril de 2025 a las 9 hs",
            "test_type": "basic_subject_query"
        },
        {
            "query": "¿Cuándo rindo el parcial de Matemática I?",
            "expected_answer": "El parcial de Matemática I se rinde el 23 de abril de 2025.",
            "test_type": "basic_subject_query"
        },
        {
            "query": "Fecha del examen de Fundamentos de Contabilidad",
            "expected_answer": "22/04/2025",
            "test_type": "basic_subject_query"
        },
        
        #tess de modalidad
        {
            "query": "¿Cuáles son los exámenes domiciliarios?",
            "expected_answer": "Exámenes domiciliarios:",
            "test_type": "modality_filter_query"
        },
        {
            "query": "¿Qué materias no tienen examen?",
            "expected_answer": "Materias sin examen:",
            "test_type": "no_exam_query"
        },
        
        #tests info específica
        {
            "query": "¿En qué aula es el examen de Fundamentos de Contabilidad?",
            "expected_answer": "HAM / H005 / H011",
            "test_type": "venue_query"
        },
        {
            "query": "¿A qué hora es el parcial de Fundamentos de la Inteligencia Artificial?",
            "expected_answer": "13hs",
            "test_type": "time_query"
        },
        
        
        {
            "query": "¿Cuándo son los parciales?",
            "expected_answer": "Fechas de parciales:",
            "test_type": "exam_type_query"
        },
        {
            "query": "¿Cuándo son los finales?",
            "expected_answer": "Fechas de finales:",
            "test_type": "exam_type_query"
        },
        {
            "query": "Fechas de recuperatorios",
            "expected_answer": "Fechas de recuperatorios:",
            "test_type": "exam_type_query"
        },
        
        {
            "query": "¿Cuándo es el examen de Física 1?",
            "expected_answer": "El examen de Física I fue el 3 de mayo de 2025 a las 17:00 hs en el aula G001",
            "test_type": "roman_arabic_conversion"
        },

        #test manejo de errores
        {
            "query": "¿Cuándo es el examen de Materia Inexistente?",
            "expected_answer": 'Parece que la materia "Materia Inexistente" no está registrada en la universidad. Si necesitas información sobre otra materia específica, por favor indícame el nombre correcto y estaré encantado de ayudarte.',
            "test_type": "error_handling"
        },
        
        {
            "query": "¿Qué modalidad tuvo el examen  de Física I?",
            "expected_answer": "El examen de Física I fue presencial.",
            "test_type": "modality_query"
        },
        
        {
            "query": "¿En qué horario se rinde el examen parcial de Fundamentos de la Inteligencia Artificial y en qué aula?",
            "expected_answer": "El examen parcial de **Fundamentos la Inteligencia Artificial** se rinde el **22 de abril de 2025** a las **13hs en el auta G001.",
            "test_type": "complex_info_query"
        }
    ]

    df = run_agent_test(TEST_CASES, output_file)
    print(f"Resultados guardados en: {output_file}")