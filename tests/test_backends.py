"""Tests for revelsMD.backends module."""

from revelsMD.backends import (
    AVAILABLE_BACKENDS,
    BACKEND_ENV_VAR,
    DEFAULT_BACKEND,
    get_backend,
)


class TestBackendConstants:
    """Tests for backend module constants."""

    def test_default_backend_is_numba(self):
        """Default backend should be numba."""
        assert DEFAULT_BACKEND == 'numba'

    def test_available_backends_contains_numpy_and_numba(self):
        """Available backends should include numpy and numba."""
        assert 'numpy' in AVAILABLE_BACKENDS
        assert 'numba' in AVAILABLE_BACKENDS

    def test_available_backends_is_frozenset(self):
        """Available backends should be immutable."""
        assert isinstance(AVAILABLE_BACKENDS, frozenset)

    def test_backend_env_var_name(self):
        """Environment variable name should be REVELSMD_BACKEND."""
        assert BACKEND_ENV_VAR == 'REVELSMD_BACKEND'


class TestGetBackend:
    """Tests for get_backend() function."""

    def test_default_when_env_unset(self, monkeypatch):
        """Returns default backend when environment variable is unset."""
        monkeypatch.delenv(BACKEND_ENV_VAR, raising=False)
        assert get_backend() == DEFAULT_BACKEND

    def test_numpy_backend_from_env(self, monkeypatch):
        """Returns 'numpy' when REVELSMD_BACKEND=numpy."""
        monkeypatch.setenv(BACKEND_ENV_VAR, 'numpy')
        assert get_backend() == 'numpy'

    def test_numba_backend_from_env(self, monkeypatch):
        """Returns 'numba' when REVELSMD_BACKEND=numba."""
        monkeypatch.setenv(BACKEND_ENV_VAR, 'numba')
        assert get_backend() == 'numba'

    def test_case_insensitive_uppercase(self, monkeypatch):
        """Backend selection is case-insensitive (NUMPY)."""
        monkeypatch.setenv(BACKEND_ENV_VAR, 'NUMPY')
        assert get_backend() == 'numpy'

    def test_case_insensitive_mixed_case(self, monkeypatch):
        """Backend selection is case-insensitive (NumPy)."""
        monkeypatch.setenv(BACKEND_ENV_VAR, 'NumPy')
        assert get_backend() == 'numpy'

    def test_case_insensitive_numba(self, monkeypatch):
        """Backend selection is case-insensitive (NUMBA)."""
        monkeypatch.setenv(BACKEND_ENV_VAR, 'NUMBA')
        assert get_backend() == 'numba'

    def test_empty_string_returns_empty(self, monkeypatch):
        """Empty string returns empty string (not default)."""
        monkeypatch.setenv(BACKEND_ENV_VAR, '')
        assert get_backend() == ''

    def test_invalid_value_returned_as_is(self, monkeypatch):
        """Invalid values are returned lowercased without validation."""
        monkeypatch.setenv(BACKEND_ENV_VAR, 'INVALID')
        assert get_backend() == 'invalid'

    def test_whitespace_not_stripped(self, monkeypatch):
        """Whitespace in value is preserved (lowercased)."""
        monkeypatch.setenv(BACKEND_ENV_VAR, ' numpy ')
        assert get_backend() == ' numpy '

    def test_reads_env_at_call_time(self, monkeypatch):
        """Backend is read fresh on each call, not cached."""
        monkeypatch.setenv(BACKEND_ENV_VAR, 'numpy')
        assert get_backend() == 'numpy'

        monkeypatch.setenv(BACKEND_ENV_VAR, 'numba')
        assert get_backend() == 'numba'
