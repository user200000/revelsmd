"""Tests for revelsMD.backends module."""

import subprocess
import sys


class TestBackendConstants:
    """Tests for backend module constants."""

    def test_default_backend_is_numba(self):
        """Default backend should be numba."""
        from revelsMD.backends import DEFAULT_BACKEND
        assert DEFAULT_BACKEND == 'numba'

    def test_available_backends_contains_numpy_and_numba(self):
        """Available backends should include numpy and numba."""
        from revelsMD.backends import AVAILABLE_BACKENDS
        assert 'numpy' in AVAILABLE_BACKENDS
        assert 'numba' in AVAILABLE_BACKENDS

    def test_available_backends_is_frozenset(self):
        """Available backends should be immutable."""
        from revelsMD.backends import AVAILABLE_BACKENDS
        assert isinstance(AVAILABLE_BACKENDS, frozenset)

    def test_backend_env_var_name(self):
        """Environment variable name should be REVELSMD_BACKEND."""
        from revelsMD.backends import BACKEND_ENV_VAR
        assert BACKEND_ENV_VAR == 'REVELSMD_BACKEND'


class TestGetBackend:
    """Tests for get_backend() function.

    Since the backend is resolved at import time, these tests use subprocess
    to test different environment configurations.
    """

    def _run_get_backend(self, env_value: str | None) -> subprocess.CompletedProcess:
        """Run a subprocess that imports backends and prints get_backend()."""
        import os
        env = os.environ.copy()
        if env_value is None:
            env.pop('REVELSMD_BACKEND', None)
        else:
            env['REVELSMD_BACKEND'] = env_value

        return subprocess.run(
            [sys.executable, '-c',
             'from revelsMD.backends import get_backend; print(get_backend())'],
            capture_output=True,
            text=True,
            env=env,
        )

    def test_default_when_env_unset(self):
        """Returns default backend when environment variable is unset."""
        result = self._run_get_backend(None)
        assert result.returncode == 0
        assert result.stdout.strip() == 'numba'

    def test_numpy_backend_from_env(self):
        """Returns 'numpy' when REVELSMD_BACKEND=numpy."""
        result = self._run_get_backend('numpy')
        assert result.returncode == 0
        assert result.stdout.strip() == 'numpy'

    def test_numba_backend_from_env(self):
        """Returns 'numba' when REVELSMD_BACKEND=numba."""
        result = self._run_get_backend('numba')
        assert result.returncode == 0
        assert result.stdout.strip() == 'numba'

    def test_case_insensitive_uppercase(self):
        """Backend selection is case-insensitive (NUMPY)."""
        result = self._run_get_backend('NUMPY')
        assert result.returncode == 0
        assert result.stdout.strip() == 'numpy'

    def test_case_insensitive_mixed_case(self):
        """Backend selection is case-insensitive (NumPy)."""
        result = self._run_get_backend('NumPy')
        assert result.returncode == 0
        assert result.stdout.strip() == 'numpy'

    def test_case_insensitive_numba(self):
        """Backend selection is case-insensitive (NUMBA)."""
        result = self._run_get_backend('NUMBA')
        assert result.returncode == 0
        assert result.stdout.strip() == 'numba'

    def test_whitespace_is_stripped(self):
        """Whitespace around backend value is stripped."""
        result = self._run_get_backend(' numpy ')
        assert result.returncode == 0
        assert result.stdout.strip() == 'numpy'

    def test_empty_string_uses_default(self):
        """Empty string falls back to default backend."""
        result = self._run_get_backend('')
        assert result.returncode == 0
        assert result.stdout.strip() == 'numba'

    def test_invalid_value_raises_error(self):
        """Invalid backend value raises ValueError at import time."""
        result = self._run_get_backend('invalid')
        assert result.returncode != 0
        assert 'ValueError' in result.stderr
        assert 'Invalid REVELSMD_BACKEND' in result.stderr

    def test_backend_is_cached(self):
        """Backend value is resolved once at import time."""
        # This test verifies that BACKEND is a module-level constant
        from revelsMD.backends import BACKEND, get_backend
        assert BACKEND == get_backend()
        assert isinstance(BACKEND, str)
        assert BACKEND in {'numpy', 'numba'}
