import os
import tempfile

from mind.self_modification import CodeSandbox, SelfModificationConfig


def test_sandbox_staging_does_not_touch_real_file_and_apply_blocked_by_default():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'a.txt')
        with open(p, 'w', encoding='utf-8') as f:
            f.write('hello\n')

        sb = CodeSandbox(config=SelfModificationConfig(enabled=False, project_root=td))
        real = sb.read_real(p)
        assert real == 'hello\n'

        sb.stage_write(p, 'changed\n')
        diff = sb.diff_for_path(p)
        assert '-hello' in diff
        assert '+changed' in diff

        # Real file is unchanged
        with open(p, 'r', encoding='utf-8') as f:
            assert f.read() == 'hello\n'

        prop = sb.create_proposal(title='t', rationale='r')
        token_resp = sb.request_approval(prop.proposal_id)
        token = token_resp['approval_token']

        # apply should be blocked since enabled=False
        out = sb.apply_proposal_to_filesystem(prop.proposal_id, token)
        assert out['success'] is False
        assert out['error'] == 'self_modification_disabled'


def test_sandbox_apply_requires_valid_token_even_when_enabled():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'a.txt')
        with open(p, 'w', encoding='utf-8') as f:
            f.write('hello\n')

        sb = CodeSandbox(config=SelfModificationConfig(enabled=True, project_root=td))
        sb.stage_write(p, 'changed\n')
        prop = sb.create_proposal(title='t', rationale='r')

        out = sb.apply_proposal_to_filesystem(prop.proposal_id, approval_token='bad')
        assert out['success'] is False
        assert out['error'] == 'approval_required'

        token = sb.request_approval(prop.proposal_id)['approval_token']
        out2 = sb.apply_proposal_to_filesystem(prop.proposal_id, approval_token=token)
        assert out2['success'] is True

        with open(p, 'r', encoding='utf-8') as f:
            assert f.read() == 'changed\n'
