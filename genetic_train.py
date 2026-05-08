"""
ENTRENAMIENTO DE AGENTE CMM PARA CATAN MEDIANTE ALGORITMO GENETICO
===================================================================

Este script implementa un algoritmo genetico para optimizar los parametros
de un agente inteligente que juega a Catan.

Flujo del algoritmo:
1. Se genera una poblacion inicial de cromosomas aleatorios (parametrizado).
2. Cada cromosoma se evalua jugando partidas contra oponentes variados.
3. Se calcula un fitness compuesto que mide el rendimiento.
4. Los mejores cromosomas se seleccionan, cruzan y mutan.
5. Se repite durante varias generaciones (parametrizado).
6. El mejor cromosoma encontrado se guarda para usarlo en el agente final.
"""

import os
import random
import json
import copy
import statistics
import concurrent.futures
from Managers.GameDirector import GameDirector

from Agents.AdrianHerasAgent import AdrianHerasAgent as aha
from Agents.AlexPelochoJaimeAgent import AlexPelochoJaimeAgent as apja
from Agents.CarlesZaidaAgent import CarlesZaidaAgent as cza
from Agents.CrabisaAgent import CrabisaAgent as ca
from Agents.EdoAgent import EdoAgent as ea
from Agents.PabloAleixAlexAgent import PabloAleixAlexAgent as paaa
from Agents.SigmaAgent import SigmaAgent as sa
from Agents.TristanAgent import TristanAgent as ta
from Agents.CMMAgent import CMMAgent, set_chromosome

# ============================================================================
# CONFIGURACION
# ============================================================================

# POP_SIZE: Numero de individuos (cromosomas) en cada generacion.
POP_SIZE = 24

# GENERATIONS: Numero de generaciones a evolucionar.
GENERATIONS = 18

# MATCHES_PER_EVAL: Partidas totales para evaluar cada cromosoma.
MATCHES_PER_EVAL = 36

# OPPONENTS: Lista de clases de agentes rivales durante el entrenamiento.
# Se excluye RandomAgent y otro porque son fáciles de ganar e inflan el fitness.
OPPONENTS = [aha, apja, cza, ca, ea, paaa, sa, ta]

# Pesos del fitness compuesto.
# La suma no necesita ser 1.0 (se normaliza implicitamente en el calculo).

ALPHA = 0.15    # Win Rate: ganar partidas
BETA = 0.30     # Point Diff: ganar por goleada
GAMMA = 0.20    # Top-2 Rate: quedar 1o o 2o
DELTA = 0.25    # Control Score: ejercito, carretera, ciudades, pueblos, cartas
EPSILON = 0.10  # Consistency: poca varianza en puntos

# ============================================================================
# BARAJA CICLICA DE OPONENTES
# ============================================================================
# Proposito: Garantizar que todos los oponentes aparecen uniformemente
# durante la evaluacion, pero en orden aleatorio para evitar sesgos.
#
# Funcionamiento:
# 1. Se baraja la lista de oponentes.
# 2. Se van cogiendo de 3 en 3 (los rivales de cada partida).
# 3. Cuando la lista se agota, se vuelve a barajar.
# 4. Asi, en cada ciclo completo, cada oponente aparece exactamente
#    el mismo numero de veces.

_shuffled_opponents = []   # Lista barajada actual
_opponent_index = 0        # Posicion actual en la lista

def _get_next_rivals():
    """
    Devuelve una lista de 3 clases de agente para usar como rivales
    en una partida de evaluacion.
    
    Implementa una baraja ciclica: recorre la lista de oponentes barajada
    y la rebaraja cuando se agota. Esto garantiza que:
    - Todos los oponentes aparecen el mismo numero de veces.
    - El orden es aleatorio en cada ciclo.
    - No se repite el mismo oponente en una misma partida (se cogen 3
      consecutivos de la lista barajada, que son todos distintos).
    
    Returns:
        list: 3 clases de agente para usar como rivales.
    """
    global _shuffled_opponents, _opponent_index
    
    # Si no hay suficientes rivales disponibles (quedan < 3), rebarajar
    if _opponent_index + 3 > len(_shuffled_opponents):
        _shuffled_opponents = list(OPPONENTS)
        random.shuffle(_shuffled_opponents)
        _opponent_index = 0
    
    # Coger 3 rivales consecutivos de la lista barajada
    rivals = _shuffled_opponents[_opponent_index:_opponent_index + 3]
    _opponent_index += 3
    return rivals


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def comprobar_escritura():
    """
    Verifica que se puede escribir en el directorio actual.
    
    Comprueba uno por uno todos los archivos que el script necesita crear
    durante el entrenamiento. Si alguno falla, aborta antes de empezar
    para no perder horas de entrenamiento por un problema de permisos.
    
    Returns:
        bool: True si todos los archivos se pueden crear, False en caso contrario.
    """
    print("\n[TEST] Verificando permisos de escritura...")
    archivos_necesarios = [
        "best_historic.json",
        "evolution_log.json",
        "best_chromosome.json",
        "errores_entrenamiento.log"
    ]

    for archivo in archivos_necesarios:
        try:
            with open(archivo, "w") as f:
                f.write("test")
            os.remove(archivo)
            print(f"  [OK] {archivo}")
        except:
            print(f"  [ERROR] {archivo}")
            return False
    print("  Todos los archivos OK.\n")
    return True


def random_chromosome():
    """
    Genera un cromosoma aleatorio para inicializar la poblacion.
    
    Cada gen es un float independiente en [0, 1] que controla
    un aspecto especifico del comportamiento del agente.
    
    Returns:
        dict: Cromosoma con 12 genes (5 de resource_pref + 7 parametros).
    """
    return {
        "resource_pref": [random.random() for _ in range(5)],
        "expansion_weight": random.random(),
        "risk_tolerance": random.random(),
        "road_priority": random.random(),
        "settlement_priority": random.random(),
        "city_priority": random.random(),
        "min_res_threshold": random.random(),
        "accept_threshold": random.random(),
        "offer_aggressiveness": random.random(),
        "scarce_bias": random.random(),
        "port_weight": random.random(),
        "thief_aggression": random.random(),
    }


def mutate(chromosome, prob=0.12, sigma=0.08):
    """
    Aplica mutacion gaussiana a un cromosoma.
    
    Cada gen tiene una probabilidad independiente 'prob' de ser mutado.
    La mutacion suma un valor aleatorio de una distribucion normal
    N(0, sigma) y luego recorta el resultado a [0, 1].
    
    Args:
        chromosome: Cromosoma a mutar (modificado in-place).
        prob: Probabilidad de mutar cada gen individual (default 0.12).
        sigma: Desviacion estandar de la mutacion (default 0.08).
               Con sigma=0.08, el 95% de las mutaciones estan entre -0.16 y +0.16.
    """
    for key in chromosome:
        if key == "resource_pref":
            # Cada preferencia de recurso muta independientemente
            for i in range(len(chromosome[key])):
                if random.random() < prob:
                    chromosome[key][i] += random.gauss(0, sigma)
                    chromosome[key][i] = max(0, min(1, chromosome[key][i]))
        else:
            if random.random() < prob:
                chromosome[key] += random.gauss(0, sigma)
                chromosome[key] = max(0, min(1, chromosome[key]))


def crossover(p1, p2):
    """
    Aplica cruce uniforme entre dos cromosomas padres.
    
    Cada gen se intercambia entre los padres con 50% de probabilidad,
    generando dos hijos. Este metodo mantiene la diversidad genetica
    mejor que el cruce de un punto.
    
    Args:
        p1: Primer cromosoma padre.
        p2: Segundo cromosoma padre.
    
    Returns:
        tuple: (hijo1, hijo2) - Dos nuevos cromosomas.
    """
    c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
    for key in p1:
        if random.random() < 0.5:
            c1[key], c2[key] = c2[key], c1[key]
    return c1, c2


def guardar_mejor_historico(best_overall):
    """
    Guarda el mejor cromosoma historico en un archivo JSON.
    
    Este archivo se sobrescribe cada vez que se encuentra un nuevo
    record, por lo que siempre contiene el mejor cromosoma encontrado
    hasta el momento. Es util para recuperar el entrenamiento si se
    interrumpe.
    
    Args:
        best_overall: Diccionario con chromosome, fitness, generation y details.
    
    Returns:
        str: Nombre del archivo guardado.
    """
    datos = {
        "chromosome": best_overall["chromosome"],
        "fitness": best_overall["fitness"],
        "generation": best_overall["generation"],
        "details": best_overall["details"],
    }

    with open("best_historic.json", "w") as f:
        json.dump(datos, f, indent=2)

    return "best_historic.json"


# ============================================================================
# EVALUACION
# ============================================================================

def _simulate_match(position, chromosome):
    """
    Simula UNA partida completa con el cromosoma dado.
    
    Esta funcion se ejecuta en paralelo para evaluar multiples partidas
    simultaneamente. Usa la baraja ciclica para elegir rivales variados.
    
    Args:
        position: Posicion de nuestro agente en la partida (0=J0, 1=J1, etc.).
        chromosome: Cromosoma a evaluar.
    
    Returns:
        tuple: (victory, points, rank, top2, point_diff, largest_army,
                longest_road, cities_built, settlements_built, cards_played)
        - victory: 1 si gano, 0 si no.
        - points: Puntos de victoria al final.
        - rank: Puesto final (1-4).
        - top2: 1 si quedo 1o o 2o, 0 si no.
        - point_diff: Diferencia con el 2o (o con el 1o si perdio).
        - largest_army: 1 si consiguio el ejercito mas grande.
        - longest_road: 1 si consiguio la carretera mas larga.
        - cities_built: Numero de ciudades construidas.
        - settlements_built: Numero de pueblos construidos.
        - cards_played: Numero de cartas de desarrollo jugadas.
    """
    try:
        set_chromosome(copy.deepcopy(chromosome))
        
        # Elegir 3 rivales de la baraja ciclica
        rivals = _get_next_rivals()
        # Insertar nuestro agente en la posicion correcta
        agents = rivals[:position] + [CMMAgent] + rivals[position:]
        
        gd = GameDirector(agents=agents, max_rounds=100, store_trace=False)
        trace = gd.game_start(print_outcome=False)

        # Extraer resultados de la ultima ronda y turno
        last_round = max(trace["game"].keys(), key=lambda r: int(r.split("_")[-1]))
        last_turn = max(trace["game"][last_round].keys(),
                        key=lambda t: int(t.split("_")[-1].lstrip("P")))
        vp = trace["game"][last_round][last_turn]["end_turn"]["victory_points"]

        agent_id = f"J{position}"
        points = int(vp[agent_id])
        all_points = {pid: int(pts) for pid, pts in vp.items()}

        # Determinar ganador y puesto
        winner = max(vp, key=lambda p: int(vp[p]))
        victory = 1 if winner == agent_id else 0
        ordenados = sorted(vp.items(), key=lambda item: int(item[1]), reverse=True)
        rank = next(idx for idx, (jug, _) in enumerate(ordenados, 1) if jug == agent_id)
        top2 = 1 if rank <= 2 else 0

        # Calcular diferencia de puntos
        sorted_points = sorted(all_points.values(), reverse=True)
        second_points = sorted_points[1] if len(sorted_points) > 1 else 0
        point_diff = points - second_points

        # Recorrer la traza para contar construcciones y cartas jugadas
        cities_built = settlements_built = cards_played = 0
        for round_key, round_data in trace.get("game", {}).items():
            for turn_key, turn_data in round_data.items():
                # Contar construcciones en fase de construccion
                build_list = turn_data.get("build_phase", [])
                for event in build_list:
                    if isinstance(event, dict):
                        building = event.get("building")
                        if building == "city":
                            cities_built += 1
                        elif building == "town":
                            settlements_built += 1

                # Contar cartas jugadas al inicio y final de turno
                for phase in ["start_turn", "end_turn"]:
                    phase_data = turn_data.get(phase, {})
                    for card in phase_data.get("development_card_played", []):
                        if isinstance(card, dict):
                            cards_played += 1

        # Obtener logros finales desde el GameManager
        players_data = gd.game_manager.get_players()
        largest_army = 1 if players_data[position].get('largest_army') else 0
        longest_road = 1 if players_data[position].get('longest_road') else 0

        return (victory, points, rank, top2, point_diff, largest_army, longest_road,
                cities_built, settlements_built, cards_played)

    except:
        # En caso de error, devolver valores neutros (derrota)
        return (0, 0, 4, 0, -10, 0, 0, 0, 0, 0)


def evaluate_chromosome(chromosome, show_progress=False):
    """
    Evalua un cromosoma jugando multiples partidas en paralelo.
    
    Ejecuta MATCHES_PER_EVAL partidas repartidas equitativamente entre
    las 4 posiciones (J0, J1, J2, J3) para evitar sesgo por orden de turno.
    Las partidas se ejecutan en paralelo usando ProcessPoolExecutor.
    
    Args:
        chromosome: Cromosoma a evaluar.
        show_progress: Si True, muestra "Partida X/Y" en tiempo real.
    
    Returns:
        dict: Diccionario con todas las metricas calculadas:
            - fitness: Puntuacion compuesta (0-1).
            - win_rate: Tasa de victorias (0-1).
            - top2_rate: Tasa de quedar 1o o 2o (0-1).
            - consistency: Inverso de la desviacion estandar (0-1).
            - avg_points: Puntos medios por partida.
            - avg_rank: Puesto medio (1-4).
            - avg_point_diff: Diferencia de puntos media.
            - control_score: Puntuacion de control del tablero (0-1).
    """
    workers = os.cpu_count()
    matches_per_pos = MATCHES_PER_EVAL // 4

    all_results = []
    totals = [0] * 10      # Acumuladores para las 10 metricas
    completed = 0
    total_matches = matches_per_pos * 4

    # Ejecutar partidas en paralelo
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_simulate_match, pos, chromosome)
            for pos in range(4)
            for _ in range(matches_per_pos)
        ]
        for f in concurrent.futures.as_completed(futures):
            result = f.result()
            all_results.append(result)
            for i, val in enumerate(result):
                totals[i] += val
            completed += 1
            
            if show_progress:
                print(f"\r    Partida {completed}/{total_matches}", end="", flush=True)

    if show_progress:
        print()

    total = len(all_results)
    
    # Componente 1: Win Rate (ALPHA)
    win_rate = totals[0] / total
    
    # Componente 2: Point Diff normalizado (BETA)
    # Se normaliza de [-8, +8] a [0, 1]
    avg_point_diff = totals[4] / total
    point_diff_normalized = max(0.0, min(1.0, (avg_point_diff + 8) / 16))

    # Componente 3: Top-2 Rate (GAMMA)
    top2_rate = totals[3] / total

    # Componente 4: Control Score (DELTA)
    # Mide el dominio del tablero mediante logros concretos
    army_rate = totals[5] / total       # % de partidas con ejercito mas grande
    road_rate = totals[6] / total       # % de partidas con carretera mas larga
    control_score = (
        0.25 * army_rate +                                    # Ejercito: +2 puntos (se puede perder)
        0.15 * road_rate +                                    # Carretera: +2 puntos (volatil)
        0.35 * min(totals[7] / total / 4.0, 1.0) +           # Ciudades: puntos permanentes
        0.20 * min(totals[8] / total / 5.0, 1.0) +           # Pueblos: base de expansion
        0.05 * min(totals[9] / total / 6.0, 1.0)             # Cartas: solo indicador de actividad
    )

    # Componente 5: Consistencia (EPSILON)
    # Penaliza agentes que alternan partidas muy buenas y muy malas
    points_list = [r[1] for r in all_results]
    std_points = statistics.stdev(points_list) if len(points_list) > 1 else 0
    consistency = 1.0 - min(std_points / 8.0, 1.0)

    # Fitness compuesto final
    fitness = (
        ALPHA * win_rate +
        BETA * point_diff_normalized +
        GAMMA * top2_rate +
        DELTA * control_score +
        EPSILON * consistency
    )

    return {
        "fitness": fitness,
        "win_rate": win_rate,
        "top2_rate": top2_rate,
        "consistency": consistency,
        "avg_points": totals[1] / total,
        "avg_rank": totals[2] / total,
        "avg_point_diff": avg_point_diff,
        "control_score": control_score
    }


# ============================================================================
# PROGRAMA PRINCIPAL
# ============================================================================

if __name__ == '__main__':

    # Verificar permisos de escritura antes de empezar
    if not comprobar_escritura():
        exit(1)

    print("=" * 80)
    print("ENTRENAMIENTO DEL AGENTE CMM CON ALGORITMO GENETICO")
    print(f"Poblacion: {POP_SIZE} | Generaciones: {GENERATIONS}")
    print(f"Partidas por evaluacion: {MATCHES_PER_EVAL}")
    print(f"CPUs: {os.cpu_count()}")
    print(f"Rivales: {[op.__name__ for op in OPPONENTS]}")
    print(f"Fitness: {ALPHA}*Win + {BETA}*PtDiff + {GAMMA}*Top2 + {DELTA}*Ctrl + {EPSILON}*Cons")
    print("=" * 80)

    # Poblacion inicial aleatoria
    population = [random_chromosome() for _ in range(POP_SIZE)]
    log = []
    best_overall_fitness = -float('inf')
    best_overall = None

    for gen in range(GENERATIONS):
        print(f"\n{'='*60}")
        print(f"GENERACION {gen+1:2d}/{GENERATIONS}")
        print(f"{'='*60}")
        print(f"{'Ind':<4} {'Fitness':<8} {'Gana':<8} {'Top2':<8} {'PtDiff':<8} {'Control':<8} {'Estable':<8}")
        print("-" * 56)

        results = []
        for idx, chrom in enumerate(population):
            print(f"  Ind {idx}: ", end="", flush=True)
            res = evaluate_chromosome(chrom, show_progress=True)
            results.append((chrom, res))
            print(f"  -> Fit={res['fitness']:.3f} Win={res['win_rate']:.0%} "
                  f"Top2={res['top2_rate']:.0%} PtDiff={res['avg_point_diff']:+.1f} "
                  f"Ctrl={res['control_score']:.3f} Cons={res['consistency']:.2f}")

        # Ordenar por fitness descendente (el mejor primero)
        results.sort(key=lambda x: x[1]["fitness"], reverse=True)
        best_gen = results[0][1]
        avg_fit = sum(r[1]["fitness"] for r in results) / len(results)

        log.append({
            "gen": gen + 1,
            "max_fitness": best_gen["fitness"],
            "avg_fitness": avg_fit
        })

        # Guardar copia del mejor de esta generacion (siempre)
        # Util para comparar evolucion entre generaciones
        with open(f"best_gen_{gen+1:02d}.json", "w") as f:
            json.dump({
                "chromosome": copy.deepcopy(results[0][0]),
                "fitness": best_gen["fitness"],
                "generation": gen + 1,
                "details": best_gen
            }, f, indent=2)
        print(f"  [Guardado] best_gen_{gen+1:02d}.json (Fitness: {best_gen['fitness']:.4f})")

        # Actualizar y guardar mejor historico si se supero el record
        if best_gen["fitness"] > best_overall_fitness:
            best_overall_fitness = best_gen["fitness"]
            best_overall = {
                "chromosome": copy.deepcopy(results[0][0]),
                "fitness": best_gen["fitness"],
                "generation": gen + 1,
                "details": best_gen
            }
            archivo = guardar_mejor_historico(best_overall)
            print(f"  [NUEVO RECORD HISTORICO] -> {archivo}")

        print(f"\n  Mejor gen: {best_gen['fitness']:.3f} | "
              f"Historico: {best_overall_fitness:.3f} | "
              f"Media: {avg_fit:.3f}")

        # === SELECCION Y EVOLUCION ===
        
        # Seleccion por torneo de tamanyo 3:
        # Se eligen 3 individuos al azar y gana el de mayor fitness.
        # Ventaja: presion selectiva moderada, mantiene diversidad.
        def tournament():
            return max(random.sample(results, 3), key=lambda x: x[1]["fitness"])[0]

        # Elitismo: los 2 mejores pasan directamente a la siguiente generacion.
        # Esto garantiza que el fitness nunca empeore.
        new_pop = [copy.deepcopy(results[0][0]), copy.deepcopy(results[1][0])]
        
        # Rellenar el resto de la poblacion con cruce y mutacion
        while len(new_pop) < POP_SIZE:
            p1, p2 = tournament(), tournament()
            
            # 80% de probabilidad de cruce, 20% de clones
            if random.random() < 0.8:
                h1, h2 = crossover(p1, p2)
            else:
                h1, h2 = copy.deepcopy(p1), copy.deepcopy(p2)
            
            # Aplicar mutacion a ambos hijos
            mutate(h1)
            mutate(h2)
            new_pop.extend([h1, h2])
        
        # Recortar a POP_SIZE por si nos pasamos
        population = new_pop[:POP_SIZE]

    # === GUARDAR RESULTADOS FINALES ===
    print("\n" + "=" * 60)
    print("GUARDANDO RESULTADOS FINALES...")
    
    # best_chromosome.json: solo el cromosoma (para test.py)
    with open("best_chromosome.json", "w") as f:
        json.dump(best_overall["chromosome"], f, indent=2)

    # evolution_log.json: fitness por generacion (para graficas)
    with open("evolution_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print("Archivos generados:")
    print("  best_chromosome.json  (solo el cromosoma, para test.py)")
    print("  best_historic.json    (cromosoma + metricas del mejor)")
    print("  best_gen_01.json ...  (mejor de cada generacion)")
    print("  evolution_log.json    (fitness por generacion)")
    
    print(f"\nMEJOR RESULTADO FINAL:")
    print(f"  Fitness:    {best_overall_fitness:.4f}")
    print(f"  Generacion: {best_overall['generation']}")
    print(f"  Win Rate:   {best_overall['details']['win_rate']:.1%}")
    print(f"  Top2 Rate:  {best_overall['details']['top2_rate']:.1%}")
    print(f"  Point Diff: {best_overall['details']['avg_point_diff']:+.1f}")
    print("=" * 60)