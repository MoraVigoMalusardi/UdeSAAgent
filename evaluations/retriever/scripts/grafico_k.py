import matplotlib

k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
avg_LLM_judge = [0.782178, 0.93, 0.940594, 0.90099, 0.95, 0.87, 0.920792, 0.945678, 0.920792, 0.9205, 0.910891, 0.910891]
avg_BERT_precision = [0.699416, 0.709465, 0.704826, 0.705287, 0.705937, 0.672911, 0.696767, 0.702046, 0.700611, 0.70, 0.705128, 0.687878]
avg_BERT_recall = [0.775943, 0.793457, 0.792576, 0.792361, 0.794712, 0.761384, 0.78313, 0.792004, 0.793443, 0.79027, 0.796425, 0.778737]
avg_BERT_f1 = [0.734877, 0.748412, 0.745312, 0.745612, 0.747039, 0.713844, 0.736763, 0.743624, 0.74348, 0.7429, 0.747174, 0.729758]

matplotlib.use('Agg')
import matplotlib.pyplot as plt
def plot_evaluation_results():
    plt.figure(figsize=(12, 8))

    plt.plot(k_values, avg_LLM_judge, marker='o', label='LLM Judge', color='blue')

    plt.plot(k_values, avg_BERT_precision, marker='s', label='BERT Precision', color='orange')

    plt.plot(k_values, avg_BERT_recall, marker='^', label='BERT Recall', color='green')

    plt.plot(k_values, avg_BERT_f1, marker='x', label='BERT F1 Score', color='red')

    plt.xlabel('k (Número de documentos recuperados)')
    plt.ylabel('Puntuación')
    plt.xticks(k_values)
    plt.grid(True)
    plt.legend()
    
    output_path = "evaluacion_rag_plot.png"
    plt.savefig(output_path)
    print(f"Gráfico guardado en: {output_path}")

if __name__ == "__main__":
    plot_evaluation_results()
    print("Gráfico de evaluación generado exitosamente.")
