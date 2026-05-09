"""
Symbolic Primitives - Shared symbolic types
============================================
This module contains basic symbolic types used across the AGI system
to avoid circular imports.
"""

from dataclasses import dataclass, field
from typing import Any, Tuple

@dataclass(frozen=True)
class Term:
    """Discrete symbolic term for reasoning."""
    name: str
    args: Tuple[Any, ...] = field(default_factory=tuple)
    
    def __repr__(self):
        if not self.args: 
            return self.name
        return f"{self.name}({', '.join(map(str, self.args))})"
    
    def __eq__(self, other):
        if not isinstance(other, Term): 
            return False
        return self.name == other.name and self.args == other.args
    
    def __hash__(self): 
        return hash((self.name, self.args))


def is_variable(x: Any) -> bool:
    """Check if x is a logical variable (starts with '?')."""
    if isinstance(x, str) and x.startswith('?'): 
        return True
    if isinstance(x, Term) and x.name.startswith('?') and not x.args: 
        return True
    return False


def get_var_name(x: Any) -> str:
    """Extract variable name from term or string."""
    if isinstance(x, str): 
        return x
    return x.name
