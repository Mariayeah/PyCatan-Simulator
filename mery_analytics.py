import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_analytics(csv_file="mery_evolution_log.csv"):
    if not os.path.exists(csv_file):
        print(f"Error: No se encontró el archivo '{csv_file}'. Asegúrate de ejecutar primero el entrenamiento.")
        return

    print(f"Cargando datos desde {csv_file}...")
    df = pd.read_csv(csv_file)
    
    if "Generacion" not in df.columns:
        print("Error: El CSV no tiene el formato esperado (falta la columna 'Generacion').")
        return

    # ---------------------------------------------------------
    # 1. Gráfica de Evolución del Fitness (Mejor vs. Media)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    # Dibujar Mejor y Media
    plt.plot(df['Generacion'], df['Mejor_Fitness'], marker='o', linewidth=2, label='Mejor Fitness', color='dodgerblue')
    plt.plot(df['Generacion'], df['Media_Fitness'], marker='x', linewidth=2, linestyle='--', label='Media Fitness', color='darkorange')
    
    # Rellenar el área de desviación estándar para ver la dispersión de la población
    plt.fill_between(df['Generacion'], 
                     df['Media_Fitness'] - df['Desviacion_Estandar'], 
                     df['Media_Fitness'] + df['Desviacion_Estandar'], 
                     color='darkorange', alpha=0.2, label='Desviación Estándar')
    
    plt.title('Evolución del Rendimiento de Mery a lo largo de las Generaciones', fontsize=14)
    plt.xlabel('Generación', fontsize=12)
    plt.ylabel('Fitness (Victorias & Puntos de Victoria)', fontsize=12)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Guardar y mostrar
    fitness_plot_path = 'mery_fitness_evolution.png'
    plt.savefig(fitness_plot_path, dpi=300)
    print(f"Gráfica de fitness guardada en: {fitness_plot_path}")
    plt.show()

    # ---------------------------------------------------------
    # 2. Gráfica de Evolución de Genes Clave
    # ---------------------------------------------------------
    # Seleccionamos genes que representan decisiones importantes para analizar la estrategia
    genes_to_plot = ['Gen_P_city', 'Gen_P_settlement', 'Gen_W_WOOD', 'Gen_W_ORE']
    
    plt.figure(figsize=(10, 6))
    
    for gene in genes_to_plot:
        if gene in df.columns:
            # Quitamos el prefijo 'Gen_' para la leyenda
            label_name = gene.replace('Gen_', '')
            plt.plot(df['Generacion'], df[gene], marker='s', linewidth=2, label=label_name)
            
    plt.title('Evolución de Parámetros Estratégicos (Genes)', fontsize=14)
    plt.xlabel('Generación', fontsize=12)
    plt.ylabel('Peso del Gen', fontsize=12)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Guardar y mostrar
    genes_plot_path = 'mery_genes_evolution.png'
    plt.savefig(genes_plot_path, dpi=300)
    print(f"Gráfica de genes guardada en: {genes_plot_path}")
    plt.show()

if __name__ == "__main__":
    plot_analytics()
