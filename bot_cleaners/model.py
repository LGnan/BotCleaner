from mesa.model import Model
from mesa.agent import Agent
from mesa.space import MultiGrid
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector
from fractions import Fraction
import math
import numpy as np


class Celda(Agent):
    def __init__(self, unique_id, model, suciedad: bool = False):
        super().__init__(unique_id, model)
        self.sucia = suciedad


class Mueble(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)


class EstacionCarga(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.ocupada = False


class RobotLimpieza(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.sig_pos = None
        self.movimientos = 0
        self.carga = 100
        self.cargando = False

    def limpiar_una_celda(self, lista_de_celdas_sucias):
        celda_a_limpiar = self.random.choice(lista_de_celdas_sucias)
        celda_a_limpiar.sucia = False
        self.sig_pos = celda_a_limpiar.pos

    def seleccionar_nueva_pos(self, lista_de_vecinos):
        self.sig_pos = self.random.choice(lista_de_vecinos).pos

    def cargar_robot(self, cercana, posiciones_estaciones):
        rise, run, n = cercana
        x, y = self.pos
        x2, y2 = posiciones_estaciones[n]
        # Implementación nueva
        estacion = self.model.grid.get_cell_list_contents([posiciones_estaciones[n]])[0]
        if not isinstance(estacion, EstacionCarga):
            print("Validar que sea de calse estacionCarga y no clase mueble")
        else:
            if not estacion.ocupada:
                if x < x2:
                    x += 1
                elif x > x2:
                    x -= 1

                if y < y2:
                    y += 1
                elif y > y2:
                    y -= 1

                self.sig_pos = x, y
                if x == x2 and y == y2:
                    self.cargando = True
                    self.carga += 25
                    estacion.ocupada = True
                if self.carga >= 75:
                    self.cargando = False
                    estacion.ocupada = False

    def buscar_estacion(self, posiciones_estaciones):
        compaDist = []
        x, y = self.pos
        i = 0
        for estacion in posiciones_estaciones:
            x2, y2 = estacion
            distancia = math.sqrt((x2 - x) ** 2 + (y2 - y) ** 2)
            compaDist.append((abs(distancia), i))
            i += 1
        compaDist.sort()
        dist, n = compaDist[0]
        x2, y2 = posiciones_estaciones[n]
        run = x2 - x
        rise = y2 - y
        return rise, run, n

    @staticmethod
    def buscar_celdas_sucia(lista_de_vecinos):
        celdas_sucias = list()
        for vecino in lista_de_vecinos:
            if isinstance(vecino, Celda) and vecino.sucia:
                celdas_sucias.append(vecino)
        return celdas_sucias

    def step(self):
        vecinos = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False
        )

        for vecino in vecinos:
            if isinstance(vecino, (Mueble, RobotLimpieza)):
                vecinos.remove(vecino)

        celdas_sucias = self.buscar_celdas_sucia(vecinos)

        if len(celdas_sucias) == 0:
            self.seleccionar_nueva_pos(vecinos)
        else:
            self.limpiar_una_celda(celdas_sucias)
        self.advance()

    def advance(self):
        if self.pos != self.sig_pos:
            self.movimientos += 1
            if self.carga > 0:
                self.carga -= 1
            else:
                print ("Tilin")
                # No se mueve
            # Profe lo cambie aca porque se me parecion raro que se quedaran sin carga y sin siquiera jalar xd (jalar me refiero a moverse)

        posiciones_estaciones = [(5, 5), (5, 15), (15, 5), (15, 15)]
        if self.carga < 25:
            estacion_cercana = self.buscar_estacion(posiciones_estaciones)
            self.cargar_robot(estacion_cercana, posiciones_estaciones)
            if self.cargando:
                self.carga += 25
                while self.carga >= 75:
                    self.movimeintos += 1
                    self.cargando = False
                    estacion = self.model.grid.get_cell_list_contents([self.pos])[0]
                    estacion.ocupada = False
                    self.model.grid.move_agent(self, self.sig_pos)
            else:
                self.model.grid.move_agent(self, self.sig_pos)
                # if not self.cargando or self.carga > 70:
                #     self.model.grid.move_agent(self, self.sig_pos)
                #     estacion = self.model.grid.get_cell_list_contents([self.pos])[0]
                #     estacion.ocupada = False
                #     self.cargando = False


class Habitacion(Model):
    def __init__(
        self,
        M: int,
        N: int,
        num_agentes: int = 5,
        porc_celdas_sucias: float = 0.6,
        porc_muebles: float = 0.1,
        modo_pos_inicial: str = "Fija",
        estacion_carga: int = 4,
    ):
        self.num_agentes = num_agentes
        self.porc_celdas_sucias = porc_celdas_sucias
        self.porc_muebles = porc_muebles
        self.estacion_carga = estacion_carga

        self.grid = MultiGrid(M, N, False)
        self.schedule = SimultaneousActivation(self)

        posiciones_disponibles = [pos for _, pos in self.grid.coord_iter()]

        # Posicionamiento de muebles
        num_muebles = int(M * N * porc_muebles)
        posiciones_muebles = self.random.sample(posiciones_disponibles, k=num_muebles)

        # Posicionamiento de Estaciones de Carga
        num_estacionCarga = 4
        posiciones_estaciones = [(5, 5), (5, 15), (15, 5), (15, 15)]

        for id, pos in enumerate(posiciones_muebles):
            mueble = Mueble(int(f"{num_agentes}0{id}") + 1, self)
            self.grid.place_agent(mueble, pos)
            posiciones_disponibles.remove(pos)

        for id, pos in enumerate(posiciones_estaciones):
            estacion = EstacionCarga(int(f"{num_agentes}0{id}") + 1, self)
            self.grid.place_agent(estacion, pos)

        # Posicionamiento de celdas sucias
        self.num_celdas_sucias = int(M * N * porc_celdas_sucias)
        posiciones_celdas_sucias = self.random.sample(
            posiciones_disponibles, k=self.num_celdas_sucias
        )

        for id, pos in enumerate(posiciones_disponibles):
            suciedad = pos in posiciones_celdas_sucias
            celda = Celda(int(f"{num_agentes}{id}") + 1, self, suciedad)
            self.grid.place_agent(celda, pos)

        # Posicionamiento de agentes robot
        if modo_pos_inicial == "Aleatoria":
            pos_inicial_robots = self.random.sample(
                posiciones_disponibles, k=num_agentes
            )
        else:  # 'Fija'
            pos_inicial_robots = [(1, 1)] * num_agentes

        for id in range(num_agentes):
            robot = RobotLimpieza(id, self)
            self.grid.place_agent(robot, pos_inicial_robots[id])
            self.schedule.add(robot)

        self.datacollector = DataCollector(
            model_reporters={
                "Grid": get_grid,
                "Cargas": get_cargas,
                "CeldasSucias": get_sucias,
            },
        )

    def step(self):
        self.datacollector.collect(self)

        self.schedule.step()

    def todoLimpio(self):
        for content, x, y in self.grid.coord_iter():
            for obj in content:
                if isinstance(obj, Celda) and obj.sucia:
                    return False
        return True


def get_grid(model: Model) -> np.ndarray:
    grid = np.zeros((model.grid.width, model.grid.height))
    for cell in model.grid.coord_iter():
        cell_content, pos = cell
        x, y = pos
        for obj in cell_content:
            if isinstance(obj, RobotLimpieza):
                grid[x][y] = 2
            elif isinstance(obj, Celda):
                grid[x][y] = int(obj.sucia)
    return grid


def get_cargas(model: Model):
    return [(agent.unique_id, agent.carga) for agent in model.schedule.agents]


def get_sucias(model: Model) -> int:
    sum_sucias = 0
    for cell in model.grid.coord_iter():
        cell_content, pos = cell
        for obj in cell_content:
            if isinstance(obj, Celda) and obj.sucia:
                sum_sucias += 1
    return sum_sucias / model.num_celdas_sucias


def get_movimientos(agent: Agent) -> dict:
    if isinstance(agent, RobotLimpieza):
        return {agent.unique_id: agent.movimientos}
    # else:
    #    return 0
