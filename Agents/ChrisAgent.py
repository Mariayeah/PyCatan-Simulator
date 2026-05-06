# Ruta del archivo: Agents/ChrisAgent/ChrisAgent.py

import random
from Classes.Constants import *
from Classes.Materials import Materials
from Classes.TradeOffer import TradeOffer
from Interfaces.AgentInterface import AgentInterface

class ChrisAgent(AgentInterface):
    """
    Agente inteligente para Catan entrenado mediante Algoritmo Genético.
    """

    def __init__(self, agent_id):
        # Es vital llamar al constructor padre para que el simulador nos asigne nuestro ID (J0, J1, etc.)[cite: 4]
        super().__init__(agent_id)
        
        # Cargamos los genes que el entrenador nos asigne. 
        # Si no hay (porque jugamos una partida suelta), cargamos valores por defecto.
        self.genes = getattr(self.__class__, 'genes_actuales', {
            'peso_ciudad': 0.5,
            'peso_pueblo': 0.5,
            'prob_carretera': 0.5,
            'agresividad_ladron': 0.5
        })
        # =========================================================
        # CROMOSOMAS (Parámetros del Algoritmo Genético)
        # Aquí definiremos las variables que el entrenador va a mutar
        # =========================================================
        # Ejemplo:
        # self.peso_construir_ciudad = 0.8
        # self.peso_construir_pueblo = 0.5
        pass

    # =================================================================
    # MÉTODOS OBLIGATORIOS (Los que debes parametrizar y entrenar)[cite: 3]
    # =================================================================

    def on_game_start(self, board_instance):
        """
        Decide la ubicación del primer pueblo y la carretera adyacente[cite: 3].
        """
        self.board = board_instance
        
        # Lógica temporal: elegir un nodo y camino aleatorio válido para no romper el juego
        possibilities = self.board.valid_starting_nodes()
        if not possibilities:
            return -1, -1
            
        chosen_node_id = random.choice(possibilities)
        possible_roads = self.board.nodes[chosen_node_id]['adjacent']
        chosen_road_to_id = random.choice(possible_roads)

        return chosen_node_id, chosen_road_to_id

    def on_build_phase(self, board_instance):
        """
        Decide qué construir basándose en las probabilidades de sus genes.
        """
        self.board = board_instance
        
        # 1. ¿Intentamos construir una CIUDAD?
        # Comprobamos si tenemos recursos y si el "dado" genético nos dice que sí
        if self.hand.resources.has_more(BuildConstants.CITY):
            if random.random() < self.genes.get('peso_ciudad', 0.5):
                posibles_ciudades = self.board.valid_city_nodes(self.id)
                if posibles_ciudades:
                    nodo_elegido = random.choice(posibles_ciudades)
                    return {'building': BuildConstants.CITY, 'node_id': nodo_elegido}

        # 2. ¿Intentamos construir un PUEBLO?
        if self.hand.resources.has_more(BuildConstants.TOWN):
            if random.random() < self.genes.get('peso_pueblo', 0.5):
                posibles_pueblos = self.board.valid_town_nodes(self.id)
                if posibles_pueblos:
                    nodo_elegido = random.choice(posibles_pueblos)
                    return {'building': BuildConstants.TOWN, 'node_id': nodo_elegido}

        # 3. ¿Intentamos construir una CARRETERA?
        if self.hand.resources.has_more(BuildConstants.ROAD):
            if random.random() < self.genes.get('prob_carretera', 0.5):
                posibles_carreteras = self.board.valid_road_nodes(self.id)
                if posibles_carreteras:
                    carretera_elegida = random.choice(posibles_carreteras)
                    return {'building': BuildConstants.ROAD, 
                            'node_id': carretera_elegida['starting_node'], 
                            'road_to': carretera_elegida['finishing_node']}

        # Si los genes dicen que no o no hay recursos, no hace nada
        return None

    def on_moving_thief(self):
        """
        Mueve el ladrón basándose en el gen de agresividad.
        """
        # Encuentra dónde está el ladrón ahora para no dejarlo ahí
        terrain_with_thief_id = -1
        casillas_validas = []
        for terrain in self.board.terrain:
            if terrain['has_thief']:
                terrain_with_thief_id = terrain['id']
            else:
                casillas_validas.append(terrain['id'])
                
        if not casillas_validas:
            return {'terrain': -1, 'player': -1}

        # Mueve el ladrón a un sitio aleatorio
        nueva_casilla = random.choice(casillas_validas)
        
        # Decide a quién robar usando el gen de agresividad
        # (Si la agresividad es alta, podría programarse para buscar al jugador con más puntos,
        # pero por ahora robamos aleatoriamente a alguien en esa casilla o a nadie)
        
        if random.random() < self.genes.get('agresividad_ladron', 0.5):
            # Lógica muy básica: le decimos al simulador que decida a quién robar pasándole un -1
            # (El simulador de PyCatan suele auto-gestionar el robo si devuelves la casilla correcta)
            jugador_a_robar = -1 
        else:
            jugador_a_robar = -1
            
        return {'terrain': nueva_casilla, 'player': jugador_a_robar}


    # =================================================================
    # MÉTODOS OPCIONALES (Para subir nota)[cite: 3]
    # =================================================================

    def on_turn_start(self):
        """Llamado al inicio de cada turno. Útil para jugar cartas de desarrollo de caballero[cite: 3]."""
        return None

    def on_commerce_phase(self, board_instance):
        """Fase de comercio: para realizar una oferta a los demás jugadores[cite: 3]."""
        self.board = board_instance
        return None

    def on_trade_offer(self, board_instance, incoming_trade_offer=TradeOffer(), player_making_offer=int):
        """Manejo de ofertas de comercio recibidas. Evalúa si aceptar o rechazar[cite: 3]."""
        # Por defecto rechaza todo
        return False

    def on_turn_end(self):
        """Al final del turno para jugar cartas de desarrollo (como puntos de victoria)[cite: 3]."""
        return None

    def on_having_more_than_7_materials_when_thief_is_called(self):
        """Decide qué descartar cuando se saca un 7 y tienes más de 7 cartas[cite: 3]."""
        # Por defecto devolvemos la mano sin tocar, el simulador descartará de forma aleatoria si es necesario[cite: 3, 4]
        return self.hand

    def on_monopoly_card_use(self):
        """Elige el material a robar con la carta Monopolio[cite: 3]."""
        # Por defecto elige arcilla
        return MaterialConstants.CLAY 

    def on_road_building_card_use(self):
        """Construye dos carreteras gratis al usar la carta[cite: 3]."""
        # Retorno esperado: {'node_id': int, 'road_to': int, 'node_id_2': int, 'road_to_2': int}
        return None

    def on_year_of_plenty_card_use(self):
        """Elige dos materiales gratis al usar la carta de Abundancia[cite: 3]."""
        # Por defecto elige madera y arcilla
        return {'material': MaterialConstants.WOOD, 'material_2': MaterialConstants.CLAY}