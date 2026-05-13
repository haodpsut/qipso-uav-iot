from .base import OptimizerResult
from .qipso import QIPSO
from .pso import PSO
from .ga import GA
from .de import DE
from .greedy import GreedyTSP, StraightLine

OPTIMIZERS = {
    "qipso": QIPSO,
    "pso": PSO,
    "ga": GA,
    "de": DE,
    "greedy": GreedyTSP,
    "straight": StraightLine,
}
