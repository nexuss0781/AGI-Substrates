import numpy as np

from nn import Tensor, AdaptiveNorm, tensor_concat, tensor_stack


def test_tensor_softmax_routes_through_adaptive_norm_for_1d():
    x = Tensor(np.array([0.1, 0.2, -0.3], dtype=float))
    y = x.softmax(axis=0)
    assert isinstance(y, Tensor)
    assert y.data.shape == x.data.shape
    assert np.all(y.data >= 0)
    assert np.isfinite(y.data).all()
    s = float(np.sum(y.data))
    assert abs(s - 1.0) < 1e-6


def test_tensor_softmax_fallback_for_non_1d():
    x = Tensor(np.array([[0.1, 0.2], [0.3, -0.4]], dtype=float))
    y = x.softmax(axis=1)
    assert isinstance(y, Tensor)
    assert y.data.shape == x.data.shape
    row_sums = np.sum(y.data, axis=1)
    assert np.allclose(row_sums, np.ones_like(row_sums), atol=1e-6)


def test_tensor_concat_and_stack_shapes_and_backward():
    a = Tensor(np.array([1.0, 2.0]))
    b = Tensor(np.array([3.0]))

    c = tensor_concat([a, b], axis=0)
    assert c.data.shape == (3,)

    s = c.sum()
    s.backward()

    assert np.allclose(a.grad, np.ones_like(a.data))
    assert np.allclose(b.grad, np.ones_like(b.data))

    a2 = Tensor(np.array([1.0, 2.0]))
    b2 = Tensor(np.array([3.0, 4.0]))
    st = tensor_stack([a2, b2], axis=0)
    assert st.data.shape == (2, 2)

    s2 = st.sum()
    s2.backward()
    assert np.allclose(a2.grad, np.ones_like(a2.data))
    assert np.allclose(b2.grad, np.ones_like(b2.data))
