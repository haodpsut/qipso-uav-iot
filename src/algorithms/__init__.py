from .base import OptimizerResult
from .qipso import QIPSO
from .qipso_de import QIPSODE
from .pso import PSO
from .ga import GA
from .de import DE
from .gwo import GWO
from .lshade import LSHADE
from .cmaes import CMAES
from .greedy import GreedyTSP, StraightLine

OPTIMIZERS = {
    "qipso":   QIPSO,
    "qipsode": QIPSODE,
    "pso":     PSO,
    "ga":      GA,
    "de":      DE,
    "gwo":     GWO,
    "lshade":  LSHADE,
    "cmaes":   CMAES,
    "greedy":  GreedyTSP,
    "straight": StraightLine,
}
