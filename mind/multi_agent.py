from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from mind.brain import AGIMind


@dataclass
class MultiAgentConfig:
    num_agents: int = 2
    latent_dim: int = 64


class MultiAgentMind:
    """Multi-agent wrapper: shared architecture, private memory.

    Each agent is a full AGIMind instance with its own memory system.
    Shared parameters/weights can be implemented later via explicit parameter tying,
    but this provides the critical invariant: *private memory namespaces*.
    """

    def __init__(self, config: Optional[MultiAgentConfig] = None):
        self.config = config if config is not None else MultiAgentConfig()
        self.agents: Dict[str, AGIMind] = {}

        for i in range(int(max(1, self.config.num_agents))):
            agent_id = f'agent_{i}'
            self.agents[agent_id] = AGIMind(latent_dim=int(self.config.latent_dim), agent_id=agent_id)

    def get(self, agent_id: str) -> AGIMind:
        return self.agents[str(agent_id)]

    def ids(self) -> List[str]:
        return list(self.agents.keys())

    def tick(self, agent_id: str, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return self.get(agent_id).tick(*args, **kwargs)
