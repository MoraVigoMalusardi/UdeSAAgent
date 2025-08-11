import pandas as pd
import matplotlib.pyplot as plt

k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
df = pd.read_csv("evaluations/retriever/scripts/resultados/rag_retrieval_evaluation.csv")

plt.figure(figsize=(10, 6))
plt.plot(df["k"], df["recall_at_k"], marker='o', label="Recall@k")
plt.plot(df["k"], df["precision_at_k"], marker='s', label="Precision@k")
plt.plot(df["k"], df["mrr_at_k"],  marker='^', label="MRR@k")
plt.plot(df["k"], df["hit_rate_at_k"], marker='x', label="HitRate@k")

plt.ylabel("Puntuación")
plt.xlabel('k (Número de documentos recuperados)')
plt.legend()
plt.xticks(k_values)
plt.grid(True)
plt.tight_layout()
plt.show()