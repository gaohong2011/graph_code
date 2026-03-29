"""Unit tests for interaction module."""

import pytest

from graph_code.tools.interaction import (
    InteractionStore,
    get_interaction_store,
    ask_user,
    confirm_action,
)


class TestInteractionStore:
    """Tests for InteractionStore class."""

    def test_store_initial_state(self):
        """Test that store starts with empty state."""
        store = InteractionStore()
        assert store.pending_question is None
        assert store.pending_confirmation is None
        assert store.last_answer is None

    def test_store_clear(self):
        """Test that clear() resets all pending interactions."""
        store = InteractionStore()
        store.pending_question = "Test question"
        store.pending_confirmation = {"action": "test"}
        store.last_answer = "Test answer"

        store.clear()

        assert store.pending_question is None
        assert store.pending_confirmation is None
        # last_answer is not cleared by clear()
        assert store.last_answer == "Test answer"


class TestGetInteractionStore:
    """Tests for get_interaction_store function."""

    def test_returns_singleton(self):
        """Test that get_interaction_store returns the same instance."""
        store1 = get_interaction_store()
        store2 = get_interaction_store()
        assert store1 is store2


class TestAskUser:
    """Tests for ask_user function."""

    def test_ask_user_sets_pending_question(self):
        """Test that ask_user sets pending_question in store."""
        store = get_interaction_store()
        store.clear()

        result = ask_user("What is your name?")

        assert store.pending_question == "What is your name?"
        assert "PENDING_QUESTION" in result
        assert "What is your name?" in result

    def test_ask_user_overwrites_previous(self):
        """Test that ask_user overwrites previous question."""
        store = get_interaction_store()
        store.clear()

        ask_user("First question")
        ask_user("Second question")

        assert store.pending_question == "Second question"


class TestConfirmAction:
    """Tests for confirm_action function."""

    def test_confirm_action_sets_pending(self):
        """Test that confirm_action sets pending_confirmation."""
        store = get_interaction_store()
        store.clear()

        result = confirm_action("Delete file", "This will delete important.txt")

        assert store.pending_confirmation is not None
        assert store.pending_confirmation["action"] == "Delete file"
        assert store.pending_confirmation["details"] == "This will delete important.txt"
        assert "PENDING_CONFIRMATION" in result

    def test_confirm_action_without_details(self):
        """Test confirm_action with minimal parameters."""
        store = get_interaction_store()
        store.clear()

        result = confirm_action("Simple action")

        assert store.pending_confirmation["action"] == "Simple action"
        assert store.pending_confirmation["details"] == ""

    def test_confirm_action_overwrites_previous(self):
        """Test that confirm_action overwrites previous confirmation."""
        store = get_interaction_store()
        store.clear()

        confirm_action("First action", "Details 1")
        confirm_action("Second action", "Details 2")

        assert store.pending_confirmation["action"] == "Second action"
        assert store.pending_confirmation["details"] == "Details 2"
