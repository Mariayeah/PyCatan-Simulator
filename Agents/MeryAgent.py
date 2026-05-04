import random
from Interfaces.AgentInterface import AgentInterface
from Classes.Constants import TerrainConstants, HarborConstants, BuildConstants

class MeryGeneticAgent(AgentInterface):
    """
    Agente MeryGeneticAgent.
    Este agente utiliza un Algoritmo Genético (cromosoma) para tomar decisiones.
    Su personalidad y estrategias cambian dependiendo de los valores de los genes.
    """

    def __init__(self, agent_id, chromosome=None):
        super().__init__(agent_id)
        
        # Si no se proporciona un cromosoma, se asigna uno por defecto (para evitar errores en pruebas aisladas)
        if chromosome is None:
            self.chromosome = {
                # Inicio de partida: Pesos de recursos (0.0 a 1.0)
                "W_WOOD": 0.5, "W_BRICK": 0.5, "W_SHEEP": 0.5, "W_WHEAT": 0.5, "W_ORE": 0.5,
                "W_prob": 1.0,           # Importancia del número del dado (0.0 a 2.0)
                "W_port_generic": 0.3,   # Valor de un puerto genérico 3:1 (0.0 a 1.0)
                "W_port_specific": 0.4,  # Valor de un puerto específico 2:1 (0.0 a 1.0)
                "W_synergy": 0.2,        # Sinergia (no implementado en esta primera versión, pero reservado)

                # Fase de Construcción: Prioridades de acciones (0.0 a 1.0)
                # Un valor alto en P_city (ej. 0.9) hará que Mery prefiera mejorar ciudades
                # sobre expandirse con carreteras si tiene los recursos para ambas cosas.
                "P_settlement": 0.8,
                "P_city": 0.9,
                "P_road": 0.4,
                "W_dev_card_buy_threshold": 0.5,  # Propensión a comprar cartas de desarrollo
                "T_save_resources": 0.2,          # Propensión a pasar el turno sin construir para ahorrar

                # Ladrón y Cartas de Desarrollo
                # Si W_knight_priority es alto, el agente jugará la carta del Caballero rápido.
                "W_knight_priority": 0.6,
                "W_block_highest_vp": 0.8,
                "W_block_highest_production": 0.7,
                "W_target_hand_size": 0.5,

                # Comercio
                "T_offer_generosity": 0.5,
                "T_accept_margin": 0.0
            }
        else:
            self.chromosome = chromosome

    def _evaluate_node(self, node_id, board_instance):
        """
        Calcula la "Utilidad" de un nodo basado en los pesos del cromosoma.
        - Genes involucrados: W_WOOD, W_BRICK, W_SHEEP, W_WHEAT, W_ORE, W_prob, W_port_generic, W_port_specific.
        
        * Por qué cambian la personalidad:
          Si Mery tiene W_ORE = 0.9 y W_WHEAT = 0.9, será una agente muy orientada a ciudades y cartas de desarrollo.
          Si W_prob es muy alto, Mery priorizará colocarse en 6s y 8s sin importar tanto el recurso.
        """
        score = 0.0
        terrains = board_instance.nodes[node_id]['contacting_terrain']
        
        # Evaluar terrenos adyacentes
        for t_id in terrains:
            terrain = board_instance.terrain[t_id]
            t_type = terrain['terrain_type']
            prob = terrain['probability']
            
            # Mapeo de constantes de terreno a nuestras claves del cromosoma
            weight_key = {
                TerrainConstants.WOOD: "W_WOOD",
                TerrainConstants.CLAY: "W_BRICK",
                TerrainConstants.WOOL: "W_SHEEP",
                TerrainConstants.CEREAL: "W_WHEAT",
                TerrainConstants.MINERAL: "W_ORE"
            }.get(t_type, None)
            
            if weight_key:
                # La utilidad suma la probabilidad del terreno multiplicada por el peso que le damos al recurso
                score += prob * self.chromosome[weight_key] * self.chromosome["W_prob"]
        
        # Evaluar puertos
        harbor = board_instance.nodes[node_id]['harbor']
        if harbor == HarborConstants.ALL:
            score += self.chromosome["W_port_generic"]
        elif harbor != HarborConstants.NONE:
            score += self.chromosome["W_port_specific"]
            
        return score

    def on_game_start(self, board_instance):
        """
        Se llama únicamente al inicio de la partida para colocar 1 pueblo y 1 carretera.
        Mery evalúa todos los nodos válidos y elige el de mayor puntuación según su cromosoma.
        """
        self.board = board_instance
        valid_nodes = self.board.valid_starting_nodes()
        
        best_score = -float('inf')
        best_node = valid_nodes[0] if valid_nodes else 0
        
        # Búsqueda del mejor nodo
        for node_id in valid_nodes:
            score = self._evaluate_node(node_id, board_instance)
            if score > best_score:
                best_score = score
                best_node = node_id
                
        # Seleccionar una carretera aleatoria entre las adyacentes al mejor nodo
        possible_roads = self.board.nodes[best_node]['adjacent']
        road_to = possible_roads[random.randint(0, len(possible_roads) - 1)]
        
        return best_node, road_to

    def on_build_phase(self, board_instance):
        """
        Trigger para la fase de construcción.
        Mery revisa qué puede construir y compara las "Utilidades" de las acciones disponibles.
        Genes: P_settlement, P_city, P_road, W_dev_card_buy_threshold, T_save_resources.
        """
        self.board = board_instance
        options = []
        
        # Comprobar si puede construir una Ciudad
        if self.hand.resources.has_more(BuildConstants.CITY):
            valid_cities = self.board.valid_city_nodes(self.id)
            if valid_cities:
                options.append({'action': 'city', 'weight': self.chromosome['P_city'], 'node_id': valid_cities[0], 'road_to': None})
                
        # Comprobar si puede construir un Pueblo (Settlement)
        if self.hand.resources.has_more(BuildConstants.TOWN):
            valid_towns = self.board.valid_town_nodes(self.id)
            if valid_towns:
                # Mery es inteligente y evaluará cuál es el mejor nodo disponible para el pueblo
                best_town_node = max(valid_towns, key=lambda n: self._evaluate_node(n, self.board))
                options.append({'action': 'town', 'weight': self.chromosome['P_settlement'], 'node_id': best_town_node, 'road_to': None})
                
        # Comprobar si puede construir una Carretera
        if self.hand.resources.has_more(BuildConstants.ROAD):
            valid_roads = self.board.valid_road_nodes(self.id)
            if valid_roads:
                # Para simplificar, toma la primera carretera válida
                options.append({'action': 'road', 'weight': self.chromosome['P_road'], 
                                'node_id': valid_roads[0]['starting_node'], 'road_to': valid_roads[0]['finishing_node']})
                                
        # Comprobar si puede comprar Carta de Desarrollo
        if self.hand.resources.has_more(BuildConstants.CARD):
            options.append({'action': 'card', 'weight': self.chromosome['W_dev_card_buy_threshold'], 'node_id': None, 'road_to': None})
            
        # Opción siempre disponible: Guardar recursos (pasar turno)
        options.append({'action': 'pass', 'weight': self.chromosome['T_save_resources'], 'node_id': None, 'road_to': None})
        
        # Elegir la acción con mayor peso
        best_option = max(options, key=lambda x: x['weight'])
        
        if best_option['action'] == 'pass':
            return None
        elif best_option['action'] == 'city':
            return {'building': BuildConstants.CITY, 'node_id': best_option['node_id'], 'road_to': None}
        elif best_option['action'] == 'town':
            return {'building': BuildConstants.TOWN, 'node_id': best_option['node_id'], 'road_to': None}
        elif best_option['action'] == 'road':
            return {'building': BuildConstants.ROAD, 'node_id': best_option['node_id'], 'road_to': best_option['road_to']}
        elif best_option['action'] == 'card':
            return {'building': BuildConstants.CARD, 'node_id': None, 'road_to': None}
            
        return None

    def on_commerce_phase(self, board_instance=None):
        """
        Fase Activa de Comercio: Si Mery tiene abundancia de un recurso (ej. > 3) y le falta otro
        que valora mucho, propondrá un intercambio a los demás.
        """
        # Materiales a evaluar
        keys = ["W_WHEAT", "W_ORE", "W_BRICK", "W_WOOD", "W_SHEEP"]
        
        # Buscar el material que más valora y del que tiene poco
        best_needed = -1
        max_need_score = -1
        
        # Buscar el material que menos valora y del que tiene mucho
        best_give = -1
        min_value = float('inf')
        
        for i in range(5):
            amount = self.hand.get_from_id(i)
            weight = self.chromosome[keys[i]]
            
            # Necesidad = Peso del material * (1 / (cantidad + 1))
            need_score = weight / (amount + 1)
            if need_score > max_need_score and amount < 2:
                max_need_score = need_score
                best_needed = i
                
            # Excedente = Cantidad * (1 / (peso + 0.1))
            give_score = amount / (weight + 0.1)
            if amount >= 2 and give_score > min_value:
                min_value = give_score
                best_give = i
                
        # Si tiene un excedente y una necesidad clara modulada por su generosidad
        if best_give != -1 and best_needed != -1 and best_give != best_needed:
            # Cuán generosa es: si T_offer_generosity es alto, da 2 por 1. Si no, 1 por 1.
            gives_amount = 2 if self.chromosome['T_offer_generosity'] > 0.5 else 1
            if self.hand.get_from_id(best_give) >= gives_amount:
                offer = TradeOffer()
                offer.gives.add_from_id(best_give, gives_amount)
                offer.receives.add_from_id(best_needed, 1)
                return offer
                
        return None

    def on_trade_offer(self, board_instance, offer, player_id):
        """
        Evalúa una oferta entrante usando T_accept_margin.
        Si la utilidad de lo que recibe menos lo que da es > T_accept_margin, acepta.
        """
        # Constantes de Materiales (0:Cereal, 1:Mineral, 2:Clay, 3:Wood, 4:Wool)
        keys = ["W_WHEAT", "W_ORE", "W_BRICK", "W_WOOD", "W_SHEEP"]
        
        value_receives = 0.0
        for mat_id, amount in enumerate(offer.receives):
            value_receives += amount * self.chromosome[keys[mat_id]]
            
        value_gives = 0.0
        for mat_id, amount in enumerate(offer.gives):
            value_gives += amount * self.chromosome[keys[mat_id]]
            
        margin = value_receives - value_gives
        
        # Solo acepta si la ganancia supera su margen mínimo (T_accept_margin)
        if margin > self.chromosome['T_accept_margin']:
            return True
            
        return False

    def on_moving_thief(self):
        """
        Mueve al ladrón utilizando W_block_highest_vp y W_block_highest_production.
        Busca al jugador con más puntos y bloquea su terreno de mayor probabilidad.
        """
        best_terrain = -1
        best_score = -1
        
        current_thief_terrain = 0
        for terrain in self.board.terrain:
            if terrain['has_thief']:
                current_thief_terrain = terrain['id']
                break
                
        for terrain in self.board.terrain:
            if terrain['id'] == current_thief_terrain or terrain['terrain_type'] == TerrainConstants.DESERT:
                continue
                
            prob = terrain['probability']
            
            # Buscar si hay algún oponente en este terreno
            has_opponent = False
            for node_id in terrain['contacting_nodes']:
                node = self.board.nodes[node_id]
                if node['player'] != -1 and node['player'] != self.id:
                    has_opponent = True
                    break
                    
            if has_opponent:
                # Utilizamos el gen W_block_highest_production para ponderar esta decisión
                score = prob * self.chromosome['W_block_highest_production']
                if score > best_score:
                    best_score = score
                    best_terrain = terrain['id']
                    
        # Fallback si no hay oponentes o no encontró terreno válido
        if best_terrain == -1:
            rand_terrain = current_thief_terrain
            while rand_terrain == current_thief_terrain or self.board.terrain[rand_terrain]['terrain_type'] == TerrainConstants.DESERT:
                rand_terrain = random.randint(0, 18)
            best_terrain = rand_terrain

        return {'terrain': best_terrain, 'player': -1}

    # ==========================================================
    # MÉTODOS AVANZADOS (CARTAS Y DESCARTES) - PUNTOS EXTRA
    # ==========================================================
    
    def on_turn_start(self):
        """
        Uso de Cartas de Desarrollo al inicio del turno.
        Si tiene un Caballero y W_knight_priority es alto, lo juega.
        """
        for card in self.development_cards_hand.hand:
            if card.type == DevelopmentCardConstants.KNIGHT:
                if random.random() < self.chromosome['W_knight_priority']:
                    return card
        return None

    def on_turn_end(self):
        """Uso de Cartas de Desarrollo al final del turno."""
        return None

    def on_having_more_than_7_materials_when_thief_is_called(self):
        """
        Descarte inteligente al salir un 7. Mery descarta los recursos
        que menos valora según su cromosoma, en lugar de hacerlo aleatoriamente.
        """
        from Classes.Hand import Hand
        
        total_materials = self.hand.get_total()
        to_discard = total_materials // 2
        discarded_hand = Hand()
        
        # Clonamos nuestras cantidades para no modificar la mano real mientras calculamos
        temp_amounts = [self.hand.get_from_id(i) for i in range(5)]
        keys = ["W_WHEAT", "W_ORE", "W_BRICK", "W_WOOD", "W_SHEEP"]
        
        for _ in range(to_discard):
            # Encontrar el material con menor peso del que tengamos al menos 1
            min_weight = float('inf')
            mat_to_discard = -1
            
            for i in range(5):
                if temp_amounts[i] > 0:
                    weight = self.chromosome[keys[i]]
                    if weight < min_weight:
                        min_weight = weight
                        mat_to_discard = i
                        
            if mat_to_discard != -1:
                temp_amounts[mat_to_discard] -= 1
                discarded_hand.add_material(mat_to_discard, 1)
                
        return discarded_hand

    def on_monopoly_card_use(self):
        """Carta Monopolio: Elige el recurso que más valora genéticamente."""
        keys = ["W_WHEAT", "W_ORE", "W_BRICK", "W_WOOD", "W_SHEEP"]
        best_mat = 0
        best_weight = -1
        for i in range(5):
            if self.chromosome[keys[i]] > best_weight:
                best_weight = self.chromosome[keys[i]]
                best_mat = i
        return best_mat

    def on_year_of_plenty_card_use(self):
        """Año de la Abundancia: Pide los dos recursos que más valora."""
        keys = ["W_WHEAT", "W_ORE", "W_BRICK", "W_WOOD", "W_SHEEP"]
        weights = [(i, self.chromosome[keys[i]]) for i in range(5)]
        # Ordenar de mayor a menor peso
        weights.sort(key=lambda x: x[1], reverse=True)
        
        mat1 = weights[0][0]
        # Si el primero es muchísimo más importante que el segundo, pide dos del mismo
        if weights[0][1] > weights[1][1] * 1.5:
            mat2 = weights[0][0]
        else:
            mat2 = weights[1][0]
            
        return {'material': mat1, 'material_2': mat2}

    def on_road_building_card_use(self):
        """Constructor de Carreteras: Construye 2 carreteras evaluando opciones."""
        valid_roads = self.board.valid_road_nodes(self.id)
        if len(valid_roads) >= 2:
            return {
                'node_id': valid_roads[0]['starting_node'],
                'road_to': valid_roads[0]['finishing_node'],
                'node_id_2': valid_roads[1]['starting_node'],
                'road_to_2': valid_roads[1]['finishing_node']
            }
        elif len(valid_roads) == 1:
            return {
                'node_id': valid_roads[0]['starting_node'],
                'road_to': valid_roads[0]['finishing_node']
            }
        return None
