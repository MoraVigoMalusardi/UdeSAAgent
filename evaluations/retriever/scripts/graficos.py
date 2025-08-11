import pandas as pd
import matplotlib.pyplot as plt
import os

# Ruta base para los resultados
RESULTADOS_DIR = os.path.join(os.path.dirname(__file__), "resultados")

def graficar_rag():
    df = pd.read_csv(os.path.join(RESULTADOS_DIR, "evaluacion_rag.csv"))

    plt.figure(figsize=(8, 5))
    plt.plot(df["k"], df["average_Recall"], marker='o', label="Recall@k")
    plt.plot(df["k"], df["average_Precision"], marker='o', label="Precision@k")
    plt.title("Evaluación del Retrieval (RAG)")
    plt.xlabel("k")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTADOS_DIR, "grafico_rag.png"))
    plt.close()


def graficar_respuestas():
    df = pd.read_csv(os.path.join(RESULTADOS_DIR, "evaluacion_respuestas_resumen_por_k.csv"))

    plt.figure(figsize=(8, 5))
    plt.plot(df["k"], df["avg_LLMJudge"], marker='o', label="LLM Judge")
    plt.plot(df["k"], df["avg_BERT_precision"], marker='o', label="BERT Precision")
    plt.plot(df["k"], df["avg_BERT_recall"], marker='o',label="BERT Recall")
    plt.plot(df["k"], df["avg_BERT_f1"], marker='o', label="BERT F1")
    plt.title("Evaluación de la Calidad de las Respuestas")
    plt.xlabel("k")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTADOS_DIR, "grafico_respuestas.png"))
    plt.close()

if __name__ == "__main__":
    #graficar_rag()
    graficar_respuestas()
    print("Gráficos guardados en la carpeta 'resultados/'.")