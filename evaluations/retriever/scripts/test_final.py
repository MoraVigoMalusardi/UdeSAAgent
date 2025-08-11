import json
import sys
import os
from tqdm import tqdm 

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from LangGraphAgent.LG_Nodes import retriever_tool, llm_with_tools, MODEL_NAME, AZURE_DEPLOYMENT_NAME
from LangGraphAgent.LG_Agent import stream_graph_updates
from langchain.evaluation import load_evaluator
from langchain.chat_models import init_chat_model
import pandas as pd
import re
import evaluate

# Cargar métricas externas
# rouge = evaluate.load("rouge")
# bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")

# === Parte 1: Evaluación del Retrieval (Recall y Precision) ===

def evaluate_rag(eval_dataset, k_values: list[int]):
    results = []

    for k in k_values:
        recall_scores = []
        precision_scores = []

        for item in eval_dataset:
            question = item["question"]
            expected_sources = item["source_chunks"]
            retrieved_docs = retriever_tool(question)

            matched_sources = sum(
                any(expected.strip() in doc.strip() for doc in retrieved_docs[:k])
                for expected in expected_sources
            )
            matched_docs = sum(
                any(expected.strip() in doc.strip() for expected in expected_sources)
                for doc in retrieved_docs[:k]
            )
            recall = matched_sources / len(expected_sources) if expected_sources else 0
            precision = matched_docs / k if k else 0
            recall_scores.append(recall)
            precision_scores.append(precision)

        avg_recall = sum(recall_scores) / len(recall_scores)
        avg_precision = sum(precision_scores) / len(precision_scores)
        print(f"[k={k}] Recall: {avg_recall:.4f} | Precision: {avg_precision:.4f}")

        results.append({
            "k": k,
            "average_Recall": avg_recall,
            "average_Precision": avg_precision
        })

    df = pd.DataFrame(results)
    output_path = os.path.join(current_dir, "resultados", "evaluacion_rag.csv")
    df.to_csv(output_path, index=False)
    return df

# === Parte 2: Evaluación de las respuestas generadas (LLM Judge, ROUGE, BLEU) ===

def answer_questions(eval_dataset, k_list, user_id="evaluation_user"):
    results = []
    i = 0

    for k in k_list:
  
        for item in eval_dataset:
            question = item["question"]
            reference = re.sub(r'\[\d+\]', '', item["answer"]).strip()

            try:
                prediction = stream_graph_updates(user_input=question, user_id=user_id+str(i))
                pred_text = re.sub(r'\[\d+\]', '', prediction).strip()
                i += 1
            except Exception as e:
                print(f"Error con pregunta '{question}': {e}")
                pred_text = ""

    #guardar en un csv para cada k pregunta, la respuesta generada y la respuesta esperada
            results.append({
                "k": k,
                "question": question,
                "predicted_answer": pred_text,
                "reference_answer": reference
            })

            # Guardar resultados en un CSV
            df = pd.DataFrame(results)
            output_path = os.path.join(current_dir, "resultados", f"respuestas_generadas_test.csv")
            df.to_csv(output_path, index=False)


def evaluate_answers(k_list, user_id="evaluation_user"):
    llm_for_judge = init_chat_model(
        MODEL_NAME,
        azure_deployment=AZURE_DEPLOYMENT_NAME,
        model_provider="azure_openai",
        verbose=False,
    )
    llm_judge_eval = load_evaluator("cot_qa", llm=llm_for_judge, chain_kwargs={"verbose": False})

    results = []

    #leer el dataset de respuestas generadas: es un csv con las columnas: k, question, predicted_answer, reference_answer
    eval_dataset_file = os.path.join(current_dir, "resultados", "respuestas_generadas_test.csv")
    eval_dataset = pd.read_csv(eval_dataset_file)
    judge_scores = []
    bert_prec_scores = []
    bert_rec_scores  = []
    bert_f1_scores = []
    

    for index, row in eval_dataset.iterrows():
        k = row["k"]
        question = row["question"]
        reference = row["reference_answer"]
        pred_text = row["predicted_answer"]

        #JUDGE LLM 

        try:
            judge_res = llm_judge_eval.evaluate_strings(
                prediction=pred_text,
                input=question,
                reference=reference,
            )
            score_j = judge_res.get("score", 0.0)
        except:
            score_j = 0.0
        judge_scores.append(score_j)


        try:
            bs = bertscore.compute(
                predictions=[pred_text],
                references=[reference],
                lang="es"           # o el idioma correspondiente
            )
            # bs["precision"], bs["recall"], bs["f1"] son listas de longitud 1
            p = bs["precision"][0]
            r = bs["recall"][0]
            f1 = bs["f1"][0]
        except Exception:
            p, r, f1 = 0.0, 0.0, 0.0

        bert_prec_scores.append(p)
        bert_rec_scores.append(r)
        bert_f1_scores.append(f1)

        # Agregar los resultados de cada pregunta al DataFrame
        results.append({
            "k": k,
            "question": question,
            "predicted_answer": pred_text,
            "reference_answer": reference,
            "LLMJudge_score": score_j,
            "BERT_prec": p,
            "BERT_rec": r,
            "BERT_f1": f1
        })

        df = pd.DataFrame(results)
        output_path = os.path.join(current_dir, "resultados", f"evaluacion_respuestas.csv")
        df.to_csv(output_path, index=False)

    summary = df.groupby("k").agg({
        "LLMJudge_score": "mean",
        "BERT_prec": "mean",
        "BERT_rec":  "mean",
        "BERT_f1":   "mean"

    }).reset_index()


    summary = summary.rename(columns={
        "LLMJudge_score": "avg_LLMJudge",
        "BERT_prec": "avg_BERT_precision",
        "BERT_rec":  "avg_BERT_recall",
        "BERT_f1":   "avg_BERT_f1"
    })


    summary_path = os.path.join(current_dir, "resultados", "evaluacion_respuestas_resumen_por_k.csv")
    summary.to_csv(summary_path, index=False)

    return summary


# === Main ===

if __name__ == "__main__":
    resultados_dir = os.path.join(current_dir, "resultados")
    os.makedirs(resultados_dir, exist_ok=True)

    # eval_dataset_file = os.path.join(current_dir, "eval_dataset.json")
    eval_dataset_file = os.path.join(current_dir, "eval_dataset.json")
   
    k_values = [9]  

    print("\n=== Loading dataset ===")
    with open(eval_dataset_file, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)

    #print("\n=== Starting RAG evaluation ===")
    #evaluate_rag(eval_data, k_values)


    print("\n=== Starting Answer generation ===")
    answer_questions(eval_data, k_values)
    print("\n=== Evaluation completed. Results saved in 'resultados' directory. ===")

    # print("\n=== Starting Answer evaluation  ===")

    # df_eval = evaluate_answers(k_values)
    # print(df_eval.to_string(index=False))
    # print("\n=== Evaluation completed. Results saved in 'resultados' directory. ===")