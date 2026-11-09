import pytest

from fetch_models import ALL_MODELS, selected_models


def test_defaults_to_every_model():
    assert selected_models(None) == ALL_MODELS
    assert selected_models("") == ALL_MODELS


def test_selects_space_separated_subset():
    assert selected_models("turbo base") == ["turbo", "base"]


def test_rejects_unknown_models():
    with pytest.raises(ValueError, match="large-v9"):
        selected_models("turbo large-v9")
