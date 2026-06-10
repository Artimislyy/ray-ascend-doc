"""Verify backward compatibility of registration functions on Ray < 2.56.

These tests use mock to simulate the absence of
`ray.util.collective.backend_registry` (which was introduced in Ray 2.56),
ensuring that both registration functions raise RuntimeError with a clear
upgrade message instead of failing silently or crashing.
"""

from unittest.mock import patch

import pytest

from ray_ascend import register_hccl_collective_backend, register_hccl_tensor_transport


class TestBackwardCompatibility:
    """Verify registration functions raise RuntimeError on Ray < 2.56."""

    @patch.dict("sys.modules", {"ray.util.collective.backend_registry": None})
    def test_register_hccl_collective_backend_raises_on_old_ray(self):
        """register_hccl_collective_backend should raise RuntimeError when
        backend_registry module is unavailable (Ray < 2.56)."""
        with pytest.raises(RuntimeError, match="requires Ray >= 2.56"):
            register_hccl_collective_backend()

    @patch.dict("sys.modules", {"ray.util.collective.backend_registry": None})
    def test_register_hccl_tensor_transport_raises_on_old_ray(self):
        """register_hccl_tensor_transport should raise RuntimeError when
        backend_registry module is unavailable (Ray < 2.56).

        It calls register_hccl_collective_backend first, so it inherits
        the same RuntimeError behavior.
        """
        with pytest.raises(RuntimeError, match="requires Ray >= 2.56"):
            register_hccl_tensor_transport()
