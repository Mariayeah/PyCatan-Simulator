import os
import json
import csv
import concurrent.futures
from Managers.GameDirector import GameDirector

from Agents.RandomAgent import RandomAgent as ra
from Agents.AdrianHerasAgent import AdrianHerasAgent as aha
from Agents.AlexPastorAgent import AlexPastorAgent as apa
from Agents.AlexPelochoJaimeAgent import AlexPelochoJaimeAgent as apja
from Agents.CarlesZaidaAgent import CarlesZaidaAgent as cza
from Agents.CrabisaAgent import CrabisaAgent as ca
from Agents.EdoAgent import EdoAgent as ea
from Agents.PabloAleixAlexAgent import PabloAleixAlexAgent as paaa
from Agents.SigmaAgent import SigmaAgent as sa
from Agents.TristanAgent import TristanAgent as ta
from Agents.CMMAgent import CMMAgent, set_chromosome

# ------------------------------------------------------------------
# Cargar el mejor cromosoma (sin repetir mensajes)
try:
    with open("best_chromosome.json", "r") as f:
        best_chrom = json.load(f)
except FileNotFoundError:
    print("ERROR: No se encontro best_chromosome.json")
    print("Ejecuta primero: python genetic_train.py")
    exit(1)

class TrainedAgent(CMMAgent):
    def __init__(self, agent_id):
        super().__init__(agent_id, chromosome=best_chrom)

# ------------------------------------------------------------------
# Configuracion
N_MATCHES_BENCHMARK = 200
ARCHIVO_RESULTADOS = "resultados_benchmark.csv"
ARCHIVO_PARCIAL = "resultados_parcial.csv"

OPPONENTS_BENCHMARK = [
    ("RandomAgent", ra),
    ("AdrianHerasAgent", aha),
    ("AlexPastorAgent", apa),
    ("AlexPelochoJaimeAgent", apja),
    ("CarlesZaidaAgent", cza),
    ("CrabisaAgent", ca),
    ("EdoAgent", ea),
    ("PabloAleixAlexAgent", paaa),
    ("SigmaAgent", sa),
    ("TristanAgent", ta)
]

# ------------------------------------------------------------------
# FUNCION GLOBAL
def _simulate_benchmark_match(position, opponent_class, agent_class):
    """Simula una partida para el benchmark."""
    try:
        agents = [opponent_class for _ in range(4)]
        agents[position] = agent_class
        gd = GameDirector(agents=agents, max_rounds=200, store_trace=False)
        trace = gd.game_start(print_outcome=False)
        
        last_round = max(trace["game"].keys(), key=lambda r: int(r.split("_")[-1]))
        last_turn = max(trace["game"][last_round].keys(),
                        key=lambda t: int(t.split("_")[-1].lstrip("P")))
        vp = trace["game"][last_round][last_turn]["end_turn"]["victory_points"]
        agent_id = f"J{position}"
        points = int(vp[agent_id])
        winner = max(vp, key=lambda p: int(vp[p]))
        victory = 1 if winner == agent_id else 0
        ordenados = sorted(vp.items(), key=lambda item: int(item[1]), reverse=True)
        rank = next(idx for idx, (jug, _) in enumerate(ordenados, 1) if jug == agent_id)
        return (victory, points, rank)
    except:
        return (0, 0, 4)


def benchmark(opponent_class, opponent_name, num_matches, agent_class):
    """Ejecuta partidas sin traza contra un oponente."""
    workers = os.cpu_count()
    matches_per_pos = num_matches // 4
    total_matches = matches_per_pos * 4

    total_wins = total_points = total_rank = total = 0
    completed = 0
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_simulate_benchmark_match, pos, opponent_class, agent_class)
            for pos in range(4)
            for _ in range(matches_per_pos)
        ]
        for f in concurrent.futures.as_completed(futures):
            try:
                v, p, r = f.result()
                total_wins += v
                total_points += p
                total_rank += r
                total += 1
            except:
                total += 1
                total_rank += 4
            
            completed += 1
            print(f"\r    Progreso: {completed}/{total_matches} partidas", end="", flush=True)

    print()  # Nueva linea al terminar
    
    win_rate = total_wins / total if total else 0
    avg_points = total_points / total if total else 0
    avg_rank = total_rank / total if total else 4
    return win_rate, avg_points, avg_rank, total


def save_partial_results(csv_rows, archivo):
    """Guarda los resultados acumulados hasta ahora."""
    with open(archivo, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Oponente", "Victorias (%)", "Puntos medios", "Puesto medio", "Partidas"])
        for row in csv_rows:
            writer.writerow(row)


# ------------------------------------------------------------------
# PROGRAMA PRINCIPAL
if __name__ == "__main__":
    print("=" * 60)
    print("BENCHMARK DEL AGENTE ENTRENADO")
    print(f"Partidas por oponente: {N_MATCHES_BENCHMARK}")
    print(f"Oponentes: {[name for name, _ in OPPONENTS_BENCHMARK]}")
    print(f"Guardado incremental: {ARCHIVO_PARCIAL}")
    print("=" * 60)

    csv_rows = []
    
    for i, (name, opp_class) in enumerate(OPPONENTS_BENCHMARK, 1):
        print(f"\n[{i}/{len(OPPONENTS_BENCHMARK)}] Probando contra {name}...")
        
        try:
            win_rate, avg_pts, avg_rank, total = benchmark(opp_class, name, N_MATCHES_BENCHMARK, TrainedAgent)
            print(f"  Victorias: {win_rate:.2%}, Puntos: {avg_pts:.1f}, Puesto: {avg_rank:.2f} (Partidas: {total})")
            csv_rows.append([name, f"{win_rate:.2%}", f"{avg_pts:.1f}", f"{avg_rank:.2f}", total])
        except Exception as e:
            print(f"  ERROR con {name}: {e}")
            csv_rows.append([name, "ERROR", str(e), "0", "0"])
        
        save_partial_results(csv_rows, ARCHIVO_PARCIAL)
        print(f"  Resultados parciales guardados ({i}/{len(OPPONENTS_BENCHMARK)})")

    with open(ARCHIVO_RESULTADOS, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Oponente", "Victorias (%)", "Puntos medios", "Puesto medio", "Partidas"])
        for row in csv_rows:
            writer.writerow(row)
    print(f"\nResultados finales guardados en {ARCHIVO_RESULTADOS}")

    print("\n" + "=" * 60)
    print("RESUMEN FINAL")
    print("=" * 60)
    for row in csv_rows:
        print(f"  {row[0]:<25} Win: {row[1]:<10} Pts: {row[2]:<8} Rank: {row[3]:<8}")
    
    valid_rows = [r for r in csv_rows if r[1] != "ERROR"]
    if valid_rows:
        avg_win = sum(float(r[1].rstrip('%')) for r in valid_rows) / len(valid_rows)
        print(f"\n  Media de victorias: {avg_win:.1f}%")