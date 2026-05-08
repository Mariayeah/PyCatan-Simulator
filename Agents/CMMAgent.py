import random
from typing import Optional, Union, Dict, Any, List, Tuple
from Classes.Constants import *
from Classes.Materials import Materials
from Classes.TradeOffer import TradeOffer
from Interfaces.AgentInterface import AgentInterface

_current_chromosome = None

def set_chromosome(chromosome):
    """Establece el cromosoma global para todos los agentes CMMAgent."""
    global _current_chromosome
    _current_chromosome = chromosome

def get_chromosome():
    """Obtiene el cromosoma global actual."""
    return _current_chromosome

class CMMAgent(AgentInterface):
    """
    Agente inteligente para Catan basado en parámetros optimizables
    mediante un algoritmo genético.

    Cada decisión del agente está influenciada por un conjunto de
    parámetros (cromosoma), que determinan su comportamiento estratégico.
    """

    # Probabilidades de salir cada número en los dados (2d6) adaptado de otro agente
    PROB_TABLE = {2: 1/36, 3: 2/36, 4: 3/36, 5: 4/36, 6: 5/36,
                  7: 0, 8: 5/36, 9: 4/36, 10: 3/36, 11: 2/36, 12: 1/36}

    def __init__(self, agent_id, chromosome=None):
        super().__init__(agent_id)
        # Si se proporciona un cromosoma, lo usa; si no, usa el global o el por defecto
        if chromosome is not None:
            self.chromosome = chromosome
        else:
            global _current_chromosome
            self.chromosome = _current_chromosome if _current_chromosome is not None else self.default_chromosome()

    @staticmethod
    def default_chromosome() -> Dict[str, Any]:
        """Genera un cromosoma por defecto con valores equilibrados."""
        return {
            "resource_pref": [0.2, 0.2, 0.2, 0.2, 0.2],  # Preferencia por cada recurso (madera, arcilla, lana, trigo, mineral)
            "expansion_weight": 0.55,  # Peso para priorizar expansión (nodos libres adyacentes)
            "risk_tolerance": 0.45,  # Tolerancia al riesgo (diversidad de recursos)
            "road_priority": 0.45,  # Prioridad para construir carreteras
            "settlement_priority": 0.65,  # Prioridad para construir pueblos
            "city_priority": 0.35,  # Prioridad para construir ciudades
            "min_res_threshold": 0.45,  # Umbral mínimo de recursos para construir
            "accept_threshold": 0.45,  # Umbral para aceptar ofertas de comercio
            "offer_aggressiveness": 0.35,  # Agresividad al hacer ofertas de comercio
            "scarce_bias": 0.25,  # Sesgo hacia recursos escasos en el comercio
            "port_weight": 0.5,  # Peso para priorizar puertos
            "thief_aggression": 0.7,  # Agresividad al mover el ladrón
        }

    # =========================
    # INICIO DE PARTIDA
    # =========================
    def on_game_start(self, board_instance):
        """
        Método: on_game_start(self, board_instance)

        Descripción:
            Se llamará al inicio de la partida. Debe decidir la ubicación del
            primer pueblo y la carretera adyacente.

        Args:
            board_instance: Objeto que contiene la información actual del tablero.

        Returns:
            tuple: (node_id, road_to)
                - node_id (int): posición donde colocar el pueblo.
                - road_to (int): nodo al que se conecta la carretera inicial.

        Consideraciones:
            Evaluar la disponibilidad de recursos y la estrategia a largo plazo,
            buscando posiciones que maximicen la obtención de recursos clave y
            ofrezcan buenas oportunidades de expansión futura.
        """
        self.board = board_instance
        best_node, best_score = -1, -float("inf")
        res_pref = self.chromosome["resource_pref"]

        # Evalúa cada nodo válido para el primer pueblo
        for node_id in self.board.valid_starting_nodes():
            t_ids = self.board.nodes[node_id]["contacting_terrain"]
            score = 0.0
            resources = set()
            pip_sum = 0.0

            # Calcula el score basado en recursos y probabilidades
            for t_id in t_ids:
                t = self.board.terrain[t_id]
                if t["terrain_type"] == TerrainConstants.DESERT:
                    continue
                prob = self.PROB_TABLE.get(t["probability"], 0.0)
                score += res_pref[t["terrain_type"]] * prob * 36  # Pondera por preferencia y probabilidad
                pip_sum += prob * 36
                resources.add(t["terrain_type"])

            # Bonificaciones por diversidad y balance de recursos
            diversity_bonus = len(resources) * 0.8
            balance_bonus = 0.0
            if len(resources) >= 3:
                balance_bonus = 0.8
            if len(resources) == 4:
                balance_bonus = 1.2

            # Bonificación por expansión (nodos libres adyacentes)
            free = sum(1 for adj in self.board.nodes[node_id]["adjacent"]
                       if self.board.nodes[adj]["player"] == -1)
            expansion_bonus = self.chromosome["expansion_weight"] * free

            # Bonificación por puerto
            port_bonus = 0.0
            if self.board.is_coastal_node(node_id) and self.board.nodes[node_id]["harbor"] != HarborConstants.NONE:
                port_bonus = self.chromosome.get("port_weight", 0.5)
                harbor = self.board.nodes[node_id]["harbor"]
                if harbor != HarborConstants.NONE:
                    port_bonus += 0.25

            # Score total para el nodo
            total = score + diversity_bonus + balance_bonus + expansion_bonus + port_bonus
            total += min(pip_sum / 18.0, 2.0) * 0.4  # Bonificación por probabilidad total

            if total > best_score:
                best_score = total
                best_node = node_id

        # Elige una carretera aleatoria adyacente al mejor nodo
        possible_roads = self.board.nodes[best_node]["adjacent"]
        best_road = random.choice(possible_roads) if possible_roads else -1
        return best_node, best_road

    # =========================
    # INICIO DE TURNO
    # =========================
    def on_turn_start(self) -> Optional[Any]:
        """
        Método: on_turn_start(self)

        Descripción:
            Se llamará al inicio de cada turno antes de tirar los dados.
            Permite decidir si se juega una carta de desarrollo.

        Returns:
            DevelopmentCard | None:
                Carta a jugar o None si no se juega ninguna.

        Consideraciones:
            Evaluar si el uso de cartas de desarrollo mejora la situación actual.

        Restricciones:
            - Solo se puede jugar una carta de desarrollo por turno (excepto puntos de victoria).
            - No se puede jugar una carta en el mismo turno en que se compra.
        """
        # Juega una carta de caballero si tiene y su agresividad lo permite
        knight_cards = self.development_cards_hand.find_card_by_effect(DevelopmentCardConstants.KNIGHT_EFFECT)
        if not knight_cards:
            return None
        if self.chromosome.get("thief_aggression", 0.7) > 0.5:
            return knight_cards[0]
        return None

    # =========================
    # COMERCIO (OFERTA)
    # =========================
    def on_commerce_phase(self, board_instance=None) -> Optional[TradeOffer]:
        """
        Método: on_commerce_phase(self, board_instance)

        Descripción:
            Se llamará al inicio de la fase de comercio para realizar una oferta.
            Recibe una copia profunda del tablero.

        Args:
            board_instance: Estado actual del tablero (copia).

        Returns:
            TradeOffer | dict | None:
                Oferta de comercio, diccionario equivalente o None si no se realiza ninguna oferta.

        Consideraciones:
            Evaluar las necesidades de recursos, las posibles ofertas de otros jugadores
            y la conveniencia de mejorar la propia posición sin beneficiar en exceso a los oponentes.
        """
        # Solo hace oferta si su agresividad lo permite
        if random.random() > self.chromosome["offer_aggressiveness"]:
            return None

        hand = self.hand.resources
        needs = self._current_needs()
        scarce_bias = self.chromosome.get("scarce_bias", 0.25)

        # Busca recursos que tenga en exceso (3 o más)
        for res_id in range(5):
            if hand.get_from_id(res_id) >= 3:
                # Si hay sesgo hacia recursos escasos, ofrece por el que menos tiene
                if scarce_bias > 0.5:
                    need_id = min(range(5), key=lambda r: hand.get_from_id(r))
                    if need_id != res_id:
                        return TradeOffer(Materials.from_ids(res_id, 1), Materials.from_ids(need_id, 1))

                # Ofrece el recurso que tiene en exceso por el que más necesita
                for need_id in sorted(needs, key=lambda x: needs[x], reverse=True):
                    if need_id != res_id and needs[need_id] > 0:
                        return TradeOffer(Materials.from_ids(res_id, 1), Materials.from_ids(need_id, 1))

        return None

    # =========================
    # RESPUESTA A OFERTAS
    # =========================
    def on_trade_offer(self, board_instance, incoming_trade_offer: TradeOffer,
                       player_making_offer: int) -> Union[bool, TradeOffer]:
        """
        Método: on_trade_offer(self, board_instance, incoming_trade_offer, player_making_offer)

        Descripción:
            Se llamará cuando el agente recibe una oferta de comercio de otro jugador.

        Args:
            board_instance: Estado actual del tablero.
            incoming_trade_offer: Oferta recibida.
            player_making_offer (int): ID del jugador que realiza la oferta.

        Returns:
            bool | TradeOffer:
                - True: aceptar la oferta.
                - False: rechazarla.
                - TradeOffer: realizar una contraoferta.

        Consideraciones:
            Evaluar la utilidad de la oferta en función de las necesidades actuales
            y futuras del agente.
        """
        needs = self._current_needs()
        give_res = self._first_nonzero(incoming_trade_offer.gives)
        receive_res = self._first_nonzero(incoming_trade_offer.receives)
        if give_res is None or receive_res is None:
            return False

        # Calcula el beneficio de la oferta
        benefit = needs[receive_res] - needs[give_res]
        if benefit >= self.chromosome["accept_threshold"]:
            return True

        # Si es agresivo, hace una contraoferta
        if random.random() < self.chromosome["offer_aggressiveness"]:
            best_wanted = max(range(5), key=lambda r: needs[r])
            worst_unwanted = min(range(5), key=lambda r: needs[r])
            return TradeOffer(Materials.from_ids(worst_unwanted, 1),
                              Materials.from_ids(best_wanted, 1))
        return False

    # =========================
    # CONSTRUCCIÓN
    # =========================
    def on_build_phase(self, board_instance) -> Optional[Dict[str, Any]]:
        """
        Método: on_build_phase(self, board_instance)

        Descripción:
            Se llamará durante la fase de construcción para decidir qué construir.

        Args:
            board_instance: Estado actual del tablero.

        Returns:
            dict | None:
                {'building': str, 'node_id': int, 'road_to': int | None}
                o None si no se construye nada.

        Consideraciones:
            Evaluar la disponibilidad de recursos, las oportunidades de expansión
            y las posibles estrategias para bloquear a los oponentes.
        """
        self.board = board_instance
        hand = self.hand.resources

        # Usa cartas de desarrollo si son útiles
        if self.development_cards_hand.hand:
            for i, card in enumerate(self.development_cards_hand.hand):
                if card.effect == DevelopmentCardConstants.YEAR_OF_PLENTY_EFFECT and sum(hand) >= 4:
                    return self.development_cards_hand.select_card(i)
                if card.effect == DevelopmentCardConstants.ROAD_BUILDING_EFFECT and len(self.board.valid_road_nodes(self.id)) > 1:
                    return self.development_cards_hand.select_card(i)

        # No construye si no tiene suficientes recursos
        if sum(hand) < self.chromosome["min_res_threshold"] * 5:
            return None

        # Genera lista de acciones posibles con sus pesos según el cromosoma
        possible = []
        if hand.has_more(BuildConstants.CITY):
            possible.append(("city", self.chromosome["city_priority"]))
        if hand.has_more(BuildConstants.TOWN):
            possible.append(("town", self.chromosome["settlement_priority"]))
        if hand.has_more(BuildConstants.ROAD):
            possible.append(("road", self.chromosome["road_priority"]))

        if not possible:
            if hand.has_more(BuildConstants.CARD):
                return {"building": BuildConstants.CARD}
            return None

        # Elige una acción aleatoria ponderada por los pesos del cromosoma
        chosen = self._weighted_choice(possible)

        # Ejecuta la acción elegida
        if chosen == "city":
            candidates = self.board.valid_city_nodes(self.id)
            if candidates:
                best = max(candidates, key=lambda nid: self._node_score(nid))
                return {"building": BuildConstants.CITY, "node_id": best}
        elif chosen == "town":
            candidates = self.board.valid_town_nodes(self.id)
            if candidates:
                best = max(candidates, key=lambda nid: self._node_score(nid))
                return {"building": BuildConstants.TOWN, "node_id": best}
        elif chosen == "road":
            roads = self.board.valid_road_nodes(self.id)
            if roads:
                best_road = max(roads, key=lambda r: self._score_road(r))
                return {
                    "building": BuildConstants.ROAD,
                    "node_id": best_road["starting_node"],
                    "road_to": best_road["finishing_node"]
                }

        # Si no se pudo construir nada, intenta construir una carretera aleatoria o comprar una carta
        if hand.has_more(BuildConstants.ROAD):
            roads = self.board.valid_road_nodes(self.id)
            if roads:
                r = random.choice(roads)
                return {"building": BuildConstants.ROAD, "node_id": r["starting_node"], "road_to": r["finishing_node"]}
        if hand.has_more(BuildConstants.CARD):
            return {"building": BuildConstants.CARD}
        return None

    # =========================
    # FINAL DE TURNO
    # =========================
    def on_turn_end(self) -> Optional[Any]:
        """
        Método: on_turn_end(self)

        Descripción:
            Se llamará al final del turno para decidir si jugar una carta de desarrollo.

        Returns:
            DevelopmentCard | None

        Consideraciones:
            Similar a on_turn_start, pero incluyendo cartas no utilizadas anteriormente.

        Restricciones:
            - Solo se puede jugar una carta por turno.
            - No se puede jugar una carta recién comprada.
        """
        # Prioriza jugar cartas de punto de victoria
        for i, card in enumerate(self.development_cards_hand.hand):
            if card.type == DevelopmentCardConstants.VICTORY_POINT:
                return self.development_cards_hand.select_card(i)
        # Si no, juega una carta de caballero si tiene
        knight = self.development_cards_hand.find_card_by_effect(DevelopmentCardConstants.KNIGHT_EFFECT)
        if knight:
            return knight[0]
        return None

    # =========================
    # DESCARTE POR LADRÓN
    # =========================
    def on_having_more_than_7_materials_when_thief_is_called(self) -> None:
        """
        Método: on_having_more_than_7_materials_when_thief_is_called(self)

        Descripción:
            Se llamará cuando el jugador debe descartar recursos tras sacar un 7.

        Returns:
            Hand: Recursos a descartar.

        Nota:
            El simulador gestiona el descarte automáticamente (floor(n/2)).
            La mano del agente se protege mediante save/restore.

        Consideraciones:
            Elegir estratégicamente qué recursos descartar para minimizar el impacto negativo.
        """
        hand = self.hand.resources
        total = sum(hand)
        if total <= 7:
            return

        to_discard = total // 2
        needs = self._current_needs()

        # Descarta los recursos que menos necesita primero
        for res in sorted(range(5), key=lambda r: needs[r]):
            if to_discard <= 0:
                break
            available = hand.get_from_id(res)
            remove = min(available, to_discard)
            if remove > 0:
                self.hand.remove_material(res, remove)
                to_discard -= remove

    # =========================
    # MOVER LADRÓN
    # =========================
    def on_moving_thief(self) -> Dict[str, int]:
        """
        Método: on_moving_thief(self)

        Descripción:
            Se llamará cuando el ladrón debe ser movido (por sacar un 7 o usar soldado).

        Returns:
            dict:
                {'terrain': int, 'player': int}

        Restricciones:
            No se puede seleccionar al propio jugador. Si ocurre, el sistema elegirá otro automáticamente.

        Consideraciones:
            Colocar el ladrón para maximizar la ventaja propia y perjudicar a los oponentes más fuertes.
        """
        best_score = -float("inf")
        best_terrain, best_player = -1, -1
        aggression = self.chromosome.get("thief_aggression", 0.7)

        # Evalúa cada terreno para mover el ladrón
        for terrain in self.board.terrain:
            if terrain["terrain_type"] == TerrainConstants.DESERT:
                continue

            # Obtiene los jugadores afectados por este terreno
            contacting = self.board.__get_contacting_nodes__(terrain["id"])
            players_there = set()
            for nid in contacting:
                pl = self.board.nodes[nid]["player"]
                if pl != -1 and pl != self.id:
                    players_there.add(pl)

            if not players_there:
                continue

            # Calcula el score para este terreno
            prob = self.PROB_TABLE.get(terrain["probability"], 0.0)
            score = len(players_there) * 2.5 + prob * 10.0  # Más jugadores y probabilidad = mejor

            if terrain["terrain_type"] == MaterialConstants.MINERAL:
                score += 1.5  # Bonificación extra por mineral (recurso escaso y valioso)

            if terrain["probability"] in [6, 8]:  # Terrenos con probabilidad 6 u 8 son más valiosos
                score += aggression * 2.0

            if score > best_score:
                best_score = score
                best_terrain = terrain["id"]
                best_player = random.choice(list(players_there))

        return {"terrain": best_terrain, "player": best_player}

    # =========================
    # CARTA MONOPOLIO
    # =========================
    def on_monopoly_card_use(self) -> int:
        """
        Método: on_monopoly_card_use(self)

        Descripción:
            Se llamará al usar una carta de monopolio.

        Returns:
            int: ID del material seleccionado.

        Consideraciones:
            Elegir el material más beneficioso según el estado actual del juego.
        """
        needs = self._current_needs()
        return max(range(5), key=lambda r: needs[r])  # Elige el recurso que más necesita

    # =========================
    # CARTA CARRETERAS
    # =========================
    def on_road_building_card_use(self) -> Dict[str, Any]:
        """
        Método: on_road_building_card_use(self)

        Descripción:
            Se llamará al usar una carta de construcción de carreteras.

        Returns:
            dict:
                {'node_id': int, 'road_to': int, 'node_id_2': int, 'road_to_2': int}

        Consideraciones:
            Decidir ubicaciones óptimas para expandirse o bloquear a los oponentes.
        """
        roads = self.board.valid_road_nodes(self.id)
        if not roads:
            return {"node_id": -1, "road_to": -1, "node_id_2": -1, "road_to_2": -1}

        # Ordena las carreteras por su score
        scored = sorted(roads, key=lambda r: self._score_road(r), reverse=True)

        # Si hay al menos dos carreteras válidas, construye dos con alta probabilidad
        if random.random() < 0.8 and len(scored) >= 2:
            return {
                "node_id": scored[0]["starting_node"], "road_to": scored[0]["finishing_node"],
                "node_id_2": scored[1]["starting_node"], "road_to_2": scored[1]["finishing_node"]
            }
        # Si no, construye solo una
        return {
            "node_id": scored[0]["starting_node"], "road_to": scored[0]["finishing_node"],
            "node_id_2": None, "road_to_2": None
        }

    # =========================
    # CARTA AÑO DE ABUNDANCIA
    # =========================
    def on_year_of_plenty_card_use(self) -> Dict[str, int]:
        """
        Método: on_year_of_plenty_card_use(self)

        Descripción:
            Se llamará al usar una carta de año de abundancia.

        Returns:
            dict:
                {'material': int, 'material_2': int}

        Consideraciones:
            Seleccionar los materiales más necesarios para el avance en la partida.
        """
        needs = self._current_needs()
        best = sorted(range(5), key=lambda r: needs[r], reverse=True)
        return {"material": best[0], "material_2": best[1]}  # Elige los dos recursos que más necesita

    # --- Métodos auxiliares ---
    def _current_needs(self) -> Dict[int, float]:
        """Calcula las necesidades actuales de recursos según la estrategia del cromosoma."""
        needs = {r: 0.0 for r in range(5)}

        # Ajusta necesidades según prioridades de construcción
        if self.chromosome["road_priority"] > 0.3:
            needs[MaterialConstants.WOOD] += 1.0
            needs[MaterialConstants.CLAY] += 1.0
        if self.chromosome["settlement_priority"] > 0.3:
            for r in [MaterialConstants.WOOD, MaterialConstants.CLAY, MaterialConstants.WOOL, MaterialConstants.CEREAL]:
                needs[r] += 0.5
        if self.chromosome["city_priority"] > 0.3:
            needs[MaterialConstants.MINERAL] += 1.5
            needs[MaterialConstants.CEREAL] += 1.0

        return needs

    def _weighted_choice(self, actions: List[Tuple[Any, float]]) -> Any:
        """Selecciona una acción aleatoria ponderada por sus pesos."""
        total = sum(w for _, w in actions)
        if total == 0:
            return actions[0][0]
        r = random.uniform(0, total)
        accum = 0.0
        for item, w in actions:
            accum += w
            if r <= accum:
                return item
        return actions[-1][0]

    def _first_nonzero(self, materials: Materials) -> Optional[int]:
        """Encuentra el primer recurso con cantidad > 0 en un objeto Materials."""
        for i in range(5):
            if materials.get_from_id(i) > 0:
                return i
        return None

    def _node_score(self, node_id: int) -> float:
        """
        Calcula el score de un nodo para decidir dónde construir.
        Considera recursos adyacentes, diversidad, probabilidad y puertos.
        """
        res_pref = self.chromosome["resource_pref"]
        score = 0.0
        resources = set()
        pip_sum = 0.0

        for t_id in self.board.nodes[node_id]["contacting_terrain"]:
            t = self.board.terrain[t_id]
            if t["terrain_type"] == TerrainConstants.DESERT:
                continue
            prob = self.PROB_TABLE.get(t["probability"], 0.0)
            score += res_pref[t["terrain_type"]] * prob * 36
            resources.add(t["terrain_type"])
            pip_sum += prob * 36

        # Bonificación por diversidad de recursos
        score += len(resources) * 0.5 * self.chromosome.get("risk_tolerance", 0.45)
        score += min(pip_sum / 18.0, 2.0) * 0.2

        # Bonificación por puerto
        if self.board.is_coastal_node(node_id) and self.board.nodes[node_id]["harbor"] != HarborConstants.NONE:
            score += self.chromosome.get("port_weight", 0.5)

        return score

    def _score_road(self, road: Dict) -> float:
        """
        Calcula el score de una carretera para decidir dónde construir.
        Considera nodos libres adyacentes (expansión) y puertos.
        """
        to = road["finishing_node"]
        free = sum(1 for adj in self.board.nodes[to]["adjacent"] if self.board.nodes[adj]["player"] == -1)
        harbor_bonus = 0.0
        if self.board.is_coastal_node(to) and self.board.nodes[to]["harbor"] != HarborConstants.NONE:
            harbor_bonus = 2.0
        return free * self.chromosome["expansion_weight"] + harbor_bonus