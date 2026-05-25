"""Tests for ariadnepy._utils."""

import pytest
from ariadnepy._utils import append


class TestAppend:
    def test_append_to_end_by_default(self):
        assert append([1, 2, 3], [9, 9]) == [1, 2, 3, 9, 9]

    def test_append_after_index(self):
        assert append([1, 2, 3], [9, 9], after=1) == [1, 9, 9, 2, 3]

    def test_append_after_zero(self):
        assert append([1, 2, 3], 0, after=0) == [0, 1, 2, 3]

    def test_append_to_tuple(self):
        assert append((1, 2), 3, after=0) == (3, 1, 2)

    def test_append_to_string(self):
        assert append("abc", "X", after=2) == "abXc"

    def test_append_to_none(self):
        assert append(None, [1, 2]) == [1, 2]

    def test_append_single_value_not_expanded(self):
        assert append([1, 2], "hello") == [1, 2, "hello"]

    def test_after_clamped_below_zero(self):
        assert append([1, 2], 9, after=-99) == [9, 1, 2]

    def test_after_clamped_above_length(self):
        assert append([1, 2], 9, after=999) == [1, 2, 9]

    def test_invalid_after_raises(self):
        with pytest.raises(TypeError):
            append([1, 2], 9, after="bad")
