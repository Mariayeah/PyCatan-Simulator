import os
import csv
import json
import random
import statistics
import multiprocessing as mp
from Agents.MeryAgent import MeryGeneticAgent
from Agents.RandomAgent import RandomAgent
from Agents.SigmaAgent import SigmaAgent
from Agents.AdrianHerasAgent import AdrianHerasAgent
from Managers.GameDirector import GameDirector

# Definimos todas las claves de los genes que usa Mery
GENE_KEYS = [
    "W_WOOD", "W_BRICK", "W_SHEEP", "W_WHEAT", "W_ORE", "W_prob",
    "W_port_generic", "W_port_specific", "W_synergy",
    "P_settlement", "P_city", "P_road", "W_dev_card_buy_threshold", "T_save_resources",
    "W_knight_priority", "W_block_highest_vp", "W_block_highest_production", "W_target_hand_size",
    "T_offer_generosity", "T_accept_margin"
]

def create_chromosome():
    """Genera un individuo (cromosoma) con valores aleatorios en sus rangos correspondientes."""
    return {
        key: random.uniform(0.0, 1.0) if key not in ["W_prob", "T_accept_margin"] 
             else (random.uniform(0.0, 2.0) if key == "W_prob" else random.uniform(-1.0, 1.0))
        for key in GENE_KEYS
    }

def get_mery_class(chromosome):
    """
    Función factoría. Crea una clase que hereda de MeryGeneticAgent pero que 
    automáticamente pasa el cromosoma específico. Esto permite inyectarlo al GameDirector.
    """
    class CustomMeryAgent(MeryGeneticAgent):
        def __init__(self, agent_id):
            super().__init__(agent_id, chromosome)
    # PyCatan suele guiarse por el __name__ de la clase
    CustomMeryAgent.__name__ = "MeryGeneticAgent"
    return CustomMeryAgent

def evaluate_individual(chromosome):
    """
    Evalúa a un individuo jugando 5 partidas contra 3 RandomAgent.
    Devuelve (cromosoma, fitness).
    """
    num_games = 5
    total_vp = 0
    wins = 0
    
    mery_class = get_mery_class(chromosome)
    
    # Pool de oponentes para diversidad
    opponent_pool = [RandomAgent, SigmaAgent, AdrianHerasAgent]
    
    for _ in range(num_games):
        try:
            # Seleccionamos 3 oponentes aleatorios para cada partida
            opponents = random.choices(opponent_pool, k=3)
            agents = [mery_class] + opponents
            
            # store_trace=False acelera dramáticamente la simulación al no escribir a disco
            director = GameDirector(for_test=False, agents=agents, max_rounds=200, store_trace=False)
            trace = director.game_start(print_outcome=False)
            
            # Buscamos la última ronda y último turno para extraer los VP finales
            last_round = max(trace["game"].keys(), key=lambda r: int(r.split("_")[-1]))
            last_turn = max(trace["game"][last_round].keys(), key=lambda t: int(t.split("_")[-1].lstrip("P")))
            vp_dict = trace["game"][last_round][last_turn]["end_turn"]["victory_points"]
            
            # Mery siempre es el Jugador 0 en este setup
            mery_vp = int(vp_dict["J0"])
            total_vp += mery_vp
            
            # Comprobamos si Mery ganó la partida
            winner = max(vp_dict, key=lambda p: int(vp_dict[p]))
            if winner == "J0":
                wins += 1
                
        except Exception as e:
            # En caso de error en la partida (ej: bucle infinito abortado), la ignoramos
            pass
            
    # El fitness premia ganar fuertemente (10 puntos extra por victoria), 
    # y los VP promedio para desempatar / premiar a los que estuvieron cerca de ganar.
    fitness = (wins * 10) + (total_vp / num_games)
    return chromosome, fitness

def tournament_selection(population, fitnesses, k=3):
    """Operador de Selección: Torneo de tamaño k."""
    selected_indices = random.sample(range(len(population)), k)
    best_index = max(selected_indices, key=lambda idx: fitnesses[idx])
    return population[best_index]

def uniform_crossover(parent1, parent2):
    """Operador de Cruce: Uniform Crossover. Se mezclan de forma equilibrada."""
    child1, child2 = {}, {}
    for key in GENE_KEYS:
        if random.random() < 0.5:
            child1[key] = parent1[key]
            child2[key] = parent2[key]
        else:
            child1[key] = parent2[key]
            child2[key] = parent1[key]
    return child1, child2

def gaussian_mutation(chromosome, mutation_rate=0.1, sigma=0.1):
    """Operador de Mutación: Mutación Gaussiana ligera."""
    mutated = chromosome.copy()
    for key in GENE_KEYS:
        if random.random() < mutation_rate:
            mutated[key] += random.gauss(0, sigma)
            
            # Recortar (clip) para evitar salir de los rangos válidos
            if key == "W_prob":
                mutated[key] = max(0.0, min(2.0, mutated[key]))
            elif key == "T_accept_margin":
                mutated[key] = max(-1.0, min(1.0, mutated[key]))
            else:
                mutated[key] = max(0.0, min(1.0, mutated[key]))
    return mutated

def save_mery_logs(generation, best_fitness, mean_fitness, std_dev, best_chromosome, filename="mery_evolution_log.csv"):
    """Sistema de Registro (MeryLogs): Guarda métricas y genes en formato CSV."""
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # Escribir cabeceras en el archivo nuevo
            headers = ["Generacion", "Mejor_Fitness", "Media_Fitness", "Desviacion_Estandar"] + [f"Gen_{k}" for k in GENE_KEYS]
            writer.writerow(headers)
        
        # Escribir la fila de datos
        row = [generation, f"{best_fitness:.2f}", f"{mean_fitness:.2f}", f"{std_dev:.2f}"] + [f"{best_chromosome[k]:.4f}" for k in GENE_KEYS]
        writer.writerow(row)

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """Pequeña función para visualizar el progreso por consola."""
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total: 
        print()

def main():
    # Parámetros del Algoritmo Genético
    POPULATION_SIZE = 12  # Múltiplo de nucleos suele ser ideal (ej. 4, 8, 12...)
    GENERATIONS = 10
    
    print("=========================================")
    print(" Iniciando Entrenamiento Genético (Mery) ")
    print("=========================================")
    
    population = [create_chromosome() for _ in range(POPULATION_SIZE)]
    
    for gen in range(1, GENERATIONS + 1):
        print(f"\n[ Generación {gen}/{GENERATIONS} ] Evaluando población...")
        
        # Evaluación Paralela
        with mp.Pool(mp.cpu_count()) as pool:
            # Mapeamos evaluate_individual a nuestra población
            # En un entorno con una barra de progreso real podríamos usar imap, 
            # pero map bloquea y devuelve en orden, lo cual está bien aquí.
            results = pool.map(evaluate_individual, population)
            
        # Parseo de resultados
        population = [r[0] for r in results]
        fitnesses = [r[1] for r in results]
        
        # Estadísticas
        best_fitness = max(fitnesses)
        best_idx = fitnesses.index(best_fitness)
        best_chromosome = population[best_idx]
        mean_fitness = statistics.mean(fitnesses)
        std_dev = statistics.stdev(fitnesses) if len(fitnesses) > 1 else 0.0
        
        print(f" -> Mejor Fitness: {best_fitness:.2f} | Media: {mean_fitness:.2f} | StdDev: {std_dev:.2f}")
        print(f" -> Muestra de Genes (Mejor): P_city={best_chromosome['P_city']:.<4.2f}, W_WOOD={best_chromosome['W_WOOD']:.<4.2f}")
        
        # MeryLogs y Guardado JSON
        save_mery_logs(gen, best_fitness, mean_fitness, std_dev, best_chromosome)
        with open("best_mery.json", "w") as f:
            json.dump(best_chromosome, f, indent=4)
            
        # Terminar si es la última generación
        if gen == GENERATIONS:
            print("\nEntrenamiento Completado. Modelo final guardado en 'best_mery.json'")
            break
            
        # Generar nueva población para la siguiente iteración
        new_population = []
        new_population.append(best_chromosome) # Elitismo: guardamos a la mejor "Mery"
        
        print_progress_bar(0, POPULATION_SIZE, prefix='Reproducción:', suffix='Completado', length=30)
        while len(new_population) < POPULATION_SIZE:
            # Selección
            p1 = tournament_selection(population, fitnesses)
            p2 = tournament_selection(population, fitnesses)
            
            # Cruce Uniforme
            c1, c2 = uniform_crossover(p1, p2)
            
            # Mutación Gaussiana
            c1 = gaussian_mutation(c1)
            c2 = gaussian_mutation(c2)
            
            new_population.append(c1)
            print_progress_bar(len(new_population), POPULATION_SIZE, prefix='Reproducción:', suffix='Completado', length=30)
            if len(new_population) < POPULATION_SIZE:
                new_population.append(c2)
                print_progress_bar(len(new_population), POPULATION_SIZE, prefix='Reproducción:', suffix='Completado', length=30)
                
        population = new_population

if __name__ == '__main__':
    # Necesario para el multiprocesamiento en Windows
    mp.freeze_support()
    main()
