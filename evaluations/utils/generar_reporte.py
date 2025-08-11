import pandas as pd
import os

def generar_reporte(csv_path: str, output_path: str) -> str:
 
    df = pd.read_csv(csv_path)
    
    lines = []
    
    lines.append("=== MÉTRICAS PROMEDIO ===")
    lines.append(f"LLM Judge: {df['LLMJudge_score'].mean():.4f}")
    lines.append(f"BERT F1: {df['BERT_f1'].mean():.4f}")
    lines.append(f"BERT Precision: {df['BERT_prec'].mean():.4f}")
    lines.append(f"BERT Recall: {df['BERT_rec'].mean():.4f}")
    
    lines.append("=== ANÁLISIS POR TIPO DE TEST ===")
    for test_type in sorted(df['test_type'].unique()):
        subset = df[df['test_type'] == test_type]
        llm_judge = subset['LLMJudge_score'].mean()
        bert_f1 = subset['BERT_f1'].mean()
        lines.append(f"{test_type}: LLM_Judge={llm_judge:.3f}, BERT_F1={bert_f1:.3f}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"Reporte generado: {output_path}")
    return output_path

if __name__ == "__main__":
    csv_file = "evaluations/exams_tool/results/exams_tool_with_agent_test.csv"
    output_report = "evaluations/exams_tool/results/reporte_agent_exams.txt"
    
    if os.path.exists(csv_file):
        generar_reporte(csv_file, output_report)
    else:
        print(f"Archivo no encontrado: {csv_file}")