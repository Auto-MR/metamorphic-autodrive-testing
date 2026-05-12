"""
UserModelRegistry — session-level store for all user-uploaded models.

Streamlit reruns the script on every interaction, so uploaded model state
must live outside the normal execution flow.  This module provides a
singleton registry keyed by a stable session ID, stored in
st.session_state under the key  _user_model_registry.

Public API
----------
registry = get_registry()           # always returns the live registry
registry.add(adapter)               # register a loaded UserModelAdapter
registry.remove(name)               # delete by display_name
registry.get(name)                  # retrieve adapter by name
registry.list_models()              # [{"name", "framework", "loaded", ...}]
registry.all_adapters()             # {name: adapter}
registry.clear()
"""

from __future__ import annotations
from typing import Dict, List, Optional
from core.adapters.user_model_adapter import UserModelAdapter


_REGISTRY_KEY = "_user_model_registry"


class _Registry:
    def __init__(self):
        self._store: Dict[str, UserModelAdapter] = {}

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def add(self, adapter: UserModelAdapter) -> None:
        """Add or replace a model by its display name."""
        self._store[adapter.model_name] = adapter

    def remove(self, name: str) -> bool:
        if name in self._store:
            del self._store[name]
            return True
        return False

    def get(self, name: str) -> Optional[UserModelAdapter]:
        return self._store.get(name)

    def clear(self) -> None:
        self._store.clear()

    # ── queries ───────────────────────────────────────────────────────────────

    def list_models(self) -> List[dict]:
        return [a.info() for a in self._store.values()]

    def all_adapters(self) -> Dict[str, UserModelAdapter]:
        return dict(self._store)

    def names(self) -> List[str]:
        return list(self._store.keys())

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, name: str) -> bool:
        return name in self._store


def get_registry() -> _Registry:
    """
    Return the singleton registry from Streamlit session_state.
    Safe to call anywhere after st has been imported.
    """
    import streamlit as st
    if _REGISTRY_KEY not in st.session_state:
        st.session_state[_REGISTRY_KEY] = _Registry()
    return st.session_state[_REGISTRY_KEY]
