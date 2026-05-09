import os
import time
import uuid
import difflib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SelfModificationConfig:
    enabled: bool = False
    project_root: Optional[str] = None


@dataclass
class CodeChange:
    path: str
    new_content: str


@dataclass
class CodeChangeProposal:
    proposal_id: str
    created_at: float
    title: str
    rationale: str
    changes: List[CodeChange]


class ApprovalGate:
    def __init__(self):
        self._pending: Dict[str, str] = {}

    def request(self, proposal_id: str) -> str:
        token = uuid.uuid4().hex
        self._pending[token] = str(proposal_id)
        return token

    def verify(self, token: str, proposal_id: str) -> bool:
        return str(self._pending.get(str(token), '')) == str(proposal_id)

    def consume(self, token: str) -> None:
        try:
            self._pending.pop(str(token), None)
        except Exception:
            return


class CodeSandbox:
    def __init__(self, config: Optional[SelfModificationConfig] = None):
        self.config = config if config is not None else SelfModificationConfig()
        self.project_root = os.path.abspath(self.config.project_root) if self.config.project_root else os.path.abspath(os.getcwd())

        self._real_cache: Dict[str, str] = {}
        self._staged: Dict[str, str] = {}

        self._proposals: Dict[str, CodeChangeProposal] = {}
        self.approval = ApprovalGate()

    def _normalize_path(self, path: str) -> str:
        p = os.path.abspath(str(path))
        root = self.project_root
        if not p.startswith(root):
            raise ValueError('path_outside_project_root')
        return p

    def read_real(self, path: str) -> str:
        p = self._normalize_path(path)
        if p in self._real_cache:
            return self._real_cache[p]
        with open(p, 'r', encoding='utf-8') as f:
            txt = f.read()
        self._real_cache[p] = txt
        return txt

    def stage_write(self, path: str, new_content: str) -> None:
        p = self._normalize_path(path)
        self._staged[p] = str(new_content)

    def staged_paths(self) -> List[str]:
        return list(self._staged.keys())

    def clear_staged(self) -> None:
        self._staged = {}

    def diff_for_path(self, path: str) -> str:
        p = self._normalize_path(path)
        before = self.read_real(p).splitlines(keepends=True)
        after = self._staged.get(p, self.read_real(p)).splitlines(keepends=True)
        diff = difflib.unified_diff(before, after, fromfile=p, tofile=p, lineterm='')
        return '\n'.join(diff)

    def diff_all(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for p in self._staged.keys():
            try:
                out[p] = self.diff_for_path(p)
            except Exception:
                out[p] = ''
        return out

    def create_proposal(self, title: str, rationale: str) -> CodeChangeProposal:
        changes = [CodeChange(path=p, new_content=c) for p, c in self._staged.items()]
        proposal = CodeChangeProposal(
            proposal_id=uuid.uuid4().hex,
            created_at=time.time(),
            title=str(title),
            rationale=str(rationale),
            changes=changes,
        )
        self._proposals[proposal.proposal_id] = proposal
        return proposal

    def list_proposals(self) -> List[str]:
        return list(self._proposals.keys())

    def get_proposal(self, proposal_id: str) -> Optional[CodeChangeProposal]:
        return self._proposals.get(str(proposal_id), None)

    def render_proposal(self, proposal_id: str) -> Dict[str, Any]:
        p = self.get_proposal(proposal_id)
        if p is None:
            return {'success': False, 'error': 'proposal_not_found'}
        diffs: Dict[str, str] = {}
        for ch in p.changes:
            try:
                self.stage_write(ch.path, ch.new_content)
                diffs[str(ch.path)] = self.diff_for_path(ch.path)
            except Exception:
                diffs[str(ch.path)] = ''
        return {
            'success': True,
            'proposal_id': p.proposal_id,
            'title': p.title,
            'rationale': p.rationale,
            'created_at': float(p.created_at),
            'num_changes': int(len(p.changes)),
            'diffs': diffs,
        }

    def request_approval(self, proposal_id: str) -> Dict[str, Any]:
        p = self.get_proposal(proposal_id)
        if p is None:
            return {'success': False, 'error': 'proposal_not_found'}
        token = self.approval.request(p.proposal_id)
        return {'success': True, 'proposal_id': p.proposal_id, 'approval_token': token}

    def apply_proposal_to_filesystem(self, proposal_id: str, approval_token: str) -> Dict[str, Any]:
        if not bool(self.config.enabled):
            return {'success': False, 'error': 'self_modification_disabled'}

        p = self.get_proposal(proposal_id)
        if p is None:
            return {'success': False, 'error': 'proposal_not_found'}
        if not self.approval.verify(approval_token, p.proposal_id):
            return {'success': False, 'error': 'approval_required'}

        applied: List[str] = []
        for ch in p.changes:
            try:
                path = self._normalize_path(ch.path)
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(ch.new_content)
                applied.append(path)
            except Exception:
                continue

        self.approval.consume(approval_token)
        return {'success': True, 'applied': applied, 'num_applied': int(len(applied))}


class SelfModificationManager:
    def __init__(self, config: Optional[SelfModificationConfig] = None):
        self.config = config if config is not None else SelfModificationConfig()
        self.sandbox = CodeSandbox(config=self.config)

    def status(self) -> Dict[str, Any]:
        return {
            'enabled': bool(self.config.enabled),
            'project_root': str(self.sandbox.project_root),
            'staged': list(self.sandbox.staged_paths()),
            'proposals': list(self.sandbox.list_proposals()),
        }
