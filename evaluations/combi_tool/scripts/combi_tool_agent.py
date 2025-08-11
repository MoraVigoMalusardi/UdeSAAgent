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
    """Evalúa el agente con combi tool usando métricas del RAG."""
    
    llm_for_judge = init_chat_model(
        MODEL_NAME,
        azure_deployment=AZURE_DEPLOYMENT_NAME,
        model_provider="azure_openai",
        verbose=False,
    )
    llm_judge_eval = load_evaluator("cot_qa", llm=llm_for_judge, chain_kwargs={"verbose": False})
    
    results = []

    for case in tqdm(test_cases, desc="Evaluando agente completo con combi queries"):
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
    output_file = os.path.join(project_root, "evaluations", "combi_tool", "results", f"combi_tool_with_agent_test.csv")

    TEST_CASES = [
        {
            "query": "¿Cuáles son los horarios de la combi desde el Campus UdeSA?",
            "expected_answer": "Horarios de salida de la combi desde Campus UdeSA: 10:45, 11:20, 12:20, 12:40, 14:10, 14:50, 15:50, 16:30, 18:15, 19:10, 20:00, 20:40",
            "test_type": "basic_schedule_request"
        },
        {
            "query": "¿Cuál es la próxima combi que sale desde Campus UdeSA?",
            "expected_answer": "La próxima combi desde Campus UdeSA sale a las",
            "test_type": "next_combi_request"
        },
        {
            "query": "Horarios de combis desde Estación Victoria",
            "expected_answer": "Horarios de salida de la combi desde Estación Victoria: 09:40, 10:20, 10:30, 11:00, 11:50, 12:00, 12:35, 13:00, 13:50, 14:20, 15:15, 16:00, 18:30",
            "test_type": "basic_schedule_request"
        },
        {
            "query": "¿A qué hora sale la próxima combi desde Estación Victoria?",
            "expected_answer": "La próxima combi desde Estación Victoria sale a las",
            "test_type": "next_combi_request"
        },
        {
            "query": "¿Hay combis que salgan después de las 9 de la noche desde Campus UdeSA?",
            "expected_answer": "No hay más combis programadas para hoy desde Campus UdeSA después de",
            "test_type": "late_time_query"
        },
        {
            "query": "horarios de combi",
            "expected_answer": "No entendí desde qué parada quieres el horario. Por favor menciona 'Estación Victoria' o 'Campus UdeSA' en tu pregunta.",
            "test_type": "ambiguous_location"
        },
        {
            "query": "¿Cuándo sale la próxima combi desde la estación?",
            "expected_answer": "La próxima combi desde la *Estación Victoria* sale a las",
            "test_type": "incomplete_location"
        },
        {
            "query": "Necesito ir desde el campus hasta la estación, ¿qué horarios hay?",
            "expected_answer": "Horarios de salida de la combi desde Campus UdeSA",
            "test_type": "natural_language_request"
        },
        {
        "query": "¿En cuánto tiempo sale la próxima combi desde Estación Victoria?",
        "expected_answer": "La próxima combi desde Estación Victoria sale en",
        "test_type": "time_until_next_combi"
        }
    ]

    df = run_agent_test(TEST_CASES, output_file)
    print(f"Resultados guardados en: {output_file}")