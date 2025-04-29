"""
Core package for the MA-CJD project.

Contains the definitions for key simulation entities (Radar, Jammer)
and MARL algorithm components (MAC, Learner, Networks).

Makes core classes easily importable via `from core import Radar`, etc.
"""
# Import key classes to make them available at the package level
from .radar import Radar
from .jammer import Jammer
# from .environment import Environment # Assuming environment is in simulation/ now
from .mac import BasicMAC
from .qmix import QMixLearner
from .networks import RNNAgent, QMixer

# Define which symbols are exported when using `from core import *`
__all__ = [
    "Radar", 
    "Jammer", 
    "BasicMAC", 
    "QMixLearner", 
    "RNNAgent", 
    "QMixer"
]