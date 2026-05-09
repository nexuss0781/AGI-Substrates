"""
AGI Mind Module

Unified intelligence hub with absolute act.py integration capability.
"""

from .brain import AGIMind, get_agi_mind, get_global_mind
from .self_awareness import SelfAwarenessSystem
from .multi_agent import MultiAgentMind, MultiAgentConfig
from .self_modification import SelfModificationManager, SelfModificationConfig

__all__ = [
    'AGIMind',
    'get_agi_mind', 
    'get_global_mind',
    'SelfAwarenessSystem',
    'MultiAgentMind',
    'MultiAgentConfig',
    'SelfModificationManager',
    'SelfModificationConfig'
]
