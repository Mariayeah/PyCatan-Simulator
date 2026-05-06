# Ruta del archivo: ga_trainer.py

import random
import time
import csv
import matplotlib.pyplot as plt  # <--- Librería para generar el PNG
from Agents.RandomAgent import RandomAgent as ra
from Agents.ChrisAgent import ChrisAgent
from Managers.GameDirector import GameDirector

# =================================================================
# HIPERPARÁMETROS DEL ALGORITMO GENÉTICO
# =================================================================
POPULATION_SIZE = 10      # Número de agentes distintos por generación
GENERATIONS = 15          # Cuántas generaciones vamos a simular
MUTATION_RATE = 0.15      # Probabilidad de que un gen cambie aleatoriamente
GAMES_PER_INDIVIDUAL = 3  # Partidas que juega cada individuo

def create_individual():
    """Genera un cromosoma con parámetros (genes) aleatorios."""
    return {
        'peso_ciudad': random.uniform(0, 1),
        'peso_pueblo': random.uniform(0, 1),
        'prob_carretera': random.uniform(0, 1),
        'agresividad_ladron': random.uniform(0, 1)
    }

def evaluate_fitness(individual):
    """Evalúa el rendimiento de un agente tras disputar varias partidas[cite: 92]."""
    ChrisAgent.genes_actuales = individual 
    victorias = 0
    puntos_totales = 0
    
    for _ in range(GAMES_PER_INDIVIDUAL):
        agents = [ChrisAgent, ra, ra, ra] 
        try:
            director = GameDirector(agents=agents, max_rounds=200, store_trace=False)
            trace = director.game_start(print_outcome=False)
            
            last_round = max(trace["game"].keys(), key=lambda r: int(r.split("_")[-1]))
            last_turn = max(trace["game"][last_round].keys(), key=lambda t: int(t.split("_")[-1].lstrip("P")))
            vp = trace["game"][last_round][last_turn]["end_turn"]["victory_points"]
            
            puntos = int(vp["J0"])
            puntos_totales += puntos
            winner = max(vp, key=lambda p: int(vp[p]))
            if winner == "J0":
                victorias += 1
        except Exception:
            pass 

    return (victorias * 20) + puntos_totales

def crossover(parent1, parent2):
    """Operador de cruce: mezcla los genes de dos padres."""
    child = {}
    for key in parent1:
        child[key] = parent1[key] if random.random() < 0.5 else parent2[key]
    return child

def mutate(individual):
    """Operador de mutación: altera ligeramente los genes."""
    for key in individual:
        if random.random() < MUTATION_RATE:
            individual[key] += random.uniform(-0.2, 0.2)
            individual[key] = max(0, min(1, individual[key]))
    return individual

def main():
    print("--- INICIANDO ENTRENAMIENTO GENÉTICO ---")
    
    population = [create_individual() for _ in range(POPULATION_SIZE)]
    
    # Lista para guardar los datos de nuestro CSV y Gráfica
    historial_evolucion = [] 
    
    for gen in range(GENERATIONS):
        print(f"\nGeneración {gen + 1}/{GENERATIONS}")
        
        fitness_scores = []
        for i, ind in enumerate(population):
            score = evaluate_fitness(ind)
            fitness_scores.append((ind, score))
            
        # Ordenar de mejor a peor
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        best_individual = fitness_scores[0][0]
        best_score = fitness_scores[0][1]
        
        # Calcular media
        avg_score = sum(score for _, score in fitness_scores) / POPULATION_SIZE
        
        print(f"-> Mejor Fitness: {best_score:.2f} | Media: {avg_score:.2f}")
        
        # Guardamos el registro de esta generación
        historial_evolucion.append([gen + 1, best_score, avg_score])
        
        # Selección (Elitismo)
        parents = [ind for ind, score in fitness_scores[:POPULATION_SIZE // 2]]
        
        # Cruce y Mutación
        next_generation = parents.copy()
        while len(next_generation) < POPULATION_SIZE:
            p1, p2 = random.sample(parents, 2)
            child = crossover(p1, p2)
            child = mutate(child)
            next_generation.append(child)
            
        population = next_generation

    print("\n--- ENTRENAMIENTO COMPLETADO ---")
    print(f"Mejor individuo histórico: {best_individual}")
    
    # ==========================================
    # GUARDAR RESULTADOS EN CSV
    # ==========================================
    csv_filename = "chris_evolution_log.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Generacion", "Mejor_Fitness", "Fitness_Medio"])
        writer.writerows(historial_evolucion)
    print(f"Datos guardados en: {csv_filename}")

    # ==========================================
    # GENERAR GRÁFICA PNG
    # ==========================================
    generaciones = [row[0] for row in historial_evolucion]
    mejores = [row[1] for row in historial_evolucion]
    medias = [row[2] for row in historial_evolucion]

    plt.figure(figsize=(10, 6))
    plt.plot(generaciones, mejores, label='Mejor Fitness', marker='o', color='green')
    plt.plot(generaciones, medias, label='Fitness Medio', marker='x', color='blue')
    plt.title('Evolución del Fitness - ChrisAgent')
    plt.xlabel('Generación')
    plt.ylabel('Puntuación de Fitness')
    plt.legend()
    plt.grid(True)
    
    png_filename = "chris_fitness_evolution.png"
    plt.savefig(png_filename)
    print(f"Gráfica guardada en: {png_filename}")

if __name__ == "__main__":
    main()