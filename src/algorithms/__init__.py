from .base import OptimizerResult
from .qipso import QIPSO
from .qipso_de import QIPSODE
from .pso import PSO
from .ga import GA
from .de import DE
from .greedy import GreedyTSP, StraightLine

OPTIMIZERS = {
    "qipso": QIPSO,
    "qipsode": QIPSODE,
    "pso": PSO,
    "ga": GA,
    "de": DE,
    "greedy": GreedyTSP,
    "straight": StraightLine,
}
