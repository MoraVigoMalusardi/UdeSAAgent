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
    """Evalúa el agente con calendar tool usando métricas del RAG."""
    
    llm_for_judge = init_chat_model(
        MODEL_NAME,
        azure_deployment=AZURE_DEPLOYMENT_NAME,
        model_provider="azure_openai",
        verbose=False,
    )
    llm_judge_eval = load_evaluator("cot_qa", llm=llm_for_judge, chain_kwargs={"verbose": False})
    
    results = []

    for case in tqdm(test_cases, desc="Evaluando agente completo con calendar queries"):
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
    output_file = os.path.join(project_root, "evaluations", "calendar_tool", "results", f"calendar_tool_with_agent_test.csv")


    TEST_CASES = [
        # tests de tipos de eventos
        {
            "query": "¿Cuáles son los feriados de 2025?",
            "expected_answer": "Aquí lo que encontré: feriados",
            "test_type": "basic_holidays_request"
        },
        {
            "query": "¿Cuándo son los parciales del primer semestre?",
            "expected_answer": "parciales, semestre 1",
            "test_type": "semester_specific_request"
        },
        {
            "query": "Fechas importantes del calendario académico",
            "expected_answer": "Aquí lo que encontré:",
            "test_type": "general_calendar_request"
        },
        
        # test proximo evento
        {
            "query": "¿Cuál es el próximo feriado?",
            "expected_answer": "El próximo feriado",
            "test_type": "next_event_request"
        },
        {
            "query": "¿Cuándo es el próximo evento académico importante?",
            "expected_answer": "El próximo evento",
            "test_type": "next_academic_event"
        },
        
        # test por tipo
        {
            "query": "¿Cuándo empiezan las vacaciones de invierno?",
            "expected_answer": "vacaciones",
            "test_type": "vacation_timing"
        },
        {
            "query": "¿Cuándo son los finales del segundo semestre?",
            "expected_answer": "finales, semestre 2",
            "test_type": "finals_timing"
        },
        {
            "query": "Fecha de inicio de clases 2025",
            "expected_answer": "inicio",
            "test_type": "semester_start"
        },
        
        #tests fechas específicas
        {
            "query": "¿Hay feriados en abril?",
            "expected_answer": "abril",
            "test_type": "month_specific_query"
        },
        {
            "query": "¿Qué eventos hay en marzo 2025?",
            "expected_answer": "marzo",
            "test_type": "month_events_query"
        },
        
        #test recuperatorios
        {
            "query": "¿Cuándo son los recuperatorios?",
            "expected_answer": "Los recuperatorios",
            "test_type": "makeup_exams_query"
        },
        
        # Tests de casos límite
        {
            "query": "calendario académico completo",
            "expected_answer": "Aquí lo que encontré:",
            "test_type": "complete_calendar_request"
        },
        
        # Test de manejo de queries ambiguas
        {
            "query": "fechas importantes",
            "expected_answer": "Aquí lo que encontré:",
            "test_type": "ambiguous_dates_query"
        }
    ]

    df = run_agent_test(TEST_CASES, output_file)
    print(f"Resultados guardados en: {output_file}")