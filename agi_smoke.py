from typing import Any, Dict

from mind.brain import AGIMind


def run_smoke(steps: int = 3) -> Dict[str, Any]:
    mind = AGIMind(latent_dim=64)
    out_last = None
    for i in range(int(max(1, steps))):
        out_last = mind.tick(
            observation=f"smoke step {i}",
            modality='text',
            reasoning_query='use world model to predict and plan',
            reasoning_context='smoke test',
            reward=0.0,
            done=False,
            learn=False,
            remember=False,
        )
    return {'ok': True, 'last': out_last}


if __name__ == '__main__':
    r = run_smoke(steps=3)
    print('SMOKE_OK', bool(r.get('ok')))
