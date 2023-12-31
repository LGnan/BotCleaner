from mesa.model import Model
from mesa.agent import Agent
from mesa.space import MultiGrid
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector
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


class RobotLimpieza(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.sig_pos = None
        self.movimientos = 0
        self.carga = 100
        self.cargando = False

    def limpiar_una_celda(self, sucio_cercano):
        celda_a_limpiar = self.random.choice(sucio_cercano)
        celda_a_limpiar.sucia = False
        self.sig_pos = celda_a_limpiar.pos

    def cargar_robot(self, cercana, posiciones_estaciones):
        rise, run, n = cercana
        x, y = self.pos
        x2, y2 = posiciones_estaciones[n]

        if x < x2:
            x += 1
        elif x > x2:
            x -= 1

        if y < y2:
            y += 1
        elif y > y2:
            y -= 1

        self.sig_pos = x, y
        if x == x2 and y == y2 and self.carga < 100:
            self.cargando = True
            if self.carga <= 75:
                self.carga += 25
                # self.model.grid.move_agent(self, self.pos)
            elif self.carga > 75 or self.carga <=100 :
                self.carga = 100
                # self.model.grid.move_agent(self, self.sig_pos)

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
    
    def ir_a_sucio(self, sucio_cercano, vecinos_sucios):
        rise, run, n = sucio_cercano
        x, y = self.pos
        x2, y2 = vecinos_sucios[n]
        if x < x2:
            x += 1
        elif x > x2:
            x -= 1

        if y < y2:
            y += 1
        elif y > y2:
            y -= 1
        self.sig_pos = x, y

    def buscar_sucio_cercano(self, vecinos_sucios):
        if not vecinos_sucios:  # Si no hay celdas sucias disponibles
            return 0, 0, 0  # O algún valor que indique que no hay celdas sucias
        suciaDist = []
        x, y = self.pos
        i = 0
        for sucio in vecinos_sucios:
            x2, y2 = sucio
            distancia = math.sqrt((x2 - x) ** 2 + (y2 - y) ** 2)
            suciaDist.append((abs(distancia), i))
            i += 1
        suciaDist.sort()
        dist, n = suciaDist[0]
        x2, y2 = vecinos_sucios[n]
        run = x - x2
        rise = y - y2
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


        posiciones_estaciones = [(5, 5), (5, 15), (15, 5), (15, 15)]

        if len(celdas_sucias) == 0:
            vecinos_sucios = [agent.pos for agent in self.model.schedule.agents if isinstance(agent, Celda) and agent.sucia]
            print(vecinos_sucios)
            if(len(vecinos_sucios) != 0):
                sucio_cercano = self.buscar_sucio_cercano(vecinos_sucios)
                self.ir_a_sucio(sucio_cercano, vecinos_sucios)
            else:
                estacion_cercana = self.buscar_estacion(posiciones_estaciones)
                self.cargar_robot(estacion_cercana, posiciones_estaciones)
                
        else:
            self.limpiar_una_celda(celdas_sucias)
        
        

    def advance(self):
        if self.pos != self.sig_pos:
            self.movimientos += 1
            self.carga -= 1

        posiciones_estaciones = [(5, 5), (5, 15), (15, 5), (15, 15)]

        if self.carga > 0:
            if self.carga < 25 or self.cargando == True:
                estacion_cercana = self.buscar_estacion(posiciones_estaciones)
                self.cargar_robot(estacion_cercana, posiciones_estaciones)
                if self.cargando == True and self.carga < 100:
                    # self.cargar_robot(estacion_cercana, posiciones_estaciones)
                    self.model.grid.move_agent(self, self.pos)
                elif self.carga > 75:
                    self.cargando = False
                    # self.cargando = False
                # if self.cargando == True:
                #     self.model.grid.move_agent(self, self.pos)
                # if not self.cargando or self.carga > 100:
                #     self.model.grid.move_agent(self, self.sig_pos)
                #     self.cargando = False
            self.model.grid.move_agent(self, self.sig_pos)  # mueve


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
        
        for id, pos in enumerate(posiciones_muebles):
            mueble = Mueble(int(f"{num_agentes}0{id}") + 1, self)
            self.grid.place_agent(mueble, pos)
            posiciones_disponibles.remove(pos)

        # Posicionamiento de Estaciones de Carga
        posiciones_estaciones = [(5, 5), (5, 15), (15, 5), (15, 15)]

       

        for id, pos in enumerate(posiciones_estaciones):
            estacion = EstacionCarga(int(f"{num_agentes}0{id}") + 1, self)
            self.grid.place_agent(estacion, pos)

        # Posicionamiento de celdas sucias
        self.num_celdas_sucias = int(M * N * porc_celdas_sucias)
        self.posiciones_celdas_sucias = self.random.sample(
            posiciones_disponibles, k=self.num_celdas_sucias
        )

        for id, pos in enumerate(posiciones_disponibles):
            suciedad = pos in self.posiciones_celdas_sucias
            celda = Celda(int(f"{num_agentes}{id}") + 1, self, suciedad)
            self.schedule.add(celda)
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
    """
    Método para la obtención de la grid y representarla en un notebook
    :param model: Modelo (entorno)
    :return: grid
    """
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
    return [
        (agent.unique_id, agent.carga)
        for agent in model.schedule.agents
        if isinstance(agent, RobotLimpieza)
    ]



def get_sucias(model: Model) -> int:
    """
    Método para determinar el número total de celdas sucias
    :param model: Modelo Mesa
    :return: número de celdas sucias
    """
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
