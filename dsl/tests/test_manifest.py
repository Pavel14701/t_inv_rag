"""Unit tests for manifest and validator.

This module tests the manifest structure and the validator that checks
indicator requests against the manifest.
"""

import pytest

from ..providers.manifest import Manifest, ManifestValidator


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.manifest
def test_valid_indicator(sample_manifest: Manifest) -> None:
    """Test that existing indicator is validated correctly."""
    v = ManifestValidator(sample_manifest)
    assert v.validate_indicator("rsi") is True
    assert v.validate_indicator("unknown") is False


@pytest.mark.unit
@pytest.mark.manifest
def test_valid_parameters(sample_manifest: Manifest) -> None:
    """Test parameter validation including type and range constraints."""
    v = ManifestValidator(sample_manifest)
    assert v.validate_parameter("rsi", "period", 14) is True
    assert v.validate_parameter("rsi", "period", 200) is False  # > max
    assert v.validate_parameter("rsi", "period", 0) is False  # < min
    assert v.validate_parameter("rsi", "period", 3.14) is False  # not integer


@pytest.mark.unit
@pytest.mark.manifest
def test_undefined_parameter_strict(sample_manifest: Manifest) -> None:
    """Test that strict mode rejects unknown parameters."""
    v = ManifestValidator(sample_manifest, allow_undefined=False)
    assert v.validate_parameter("rsi", "unknown_param", 10) is False


@pytest.mark.unit
@pytest.mark.manifest
def test_undefined_parameter_allowed(sample_manifest: Manifest) -> None:
    """Test that non-strict mode allows unknown parameters."""
    v = ManifestValidator(sample_manifest, allow_undefined=True)
    assert v.validate_parameter("rsi", "unknown_param", 10) is True


@pytest.mark.unit
@pytest.mark.manifest
def test_valid_attributes(sample_manifest: Manifest) -> None:
    """Test attribute validation."""
    v = ManifestValidator(sample_manifest)
    assert v.validate_attributes("rsi", ["value"]) is True
    assert v.validate_attributes("rsi", ["value", "signal"]) is True
    assert v.validate_attributes("rsi", ["nonexistent"]) is False
    # Empty attributes are always valid
    assert v.validate_attributes("unknown", []) is True


@pytest.mark.unit
@pytest.mark.manifest
def test_full_validation(sample_manifest: Manifest) -> None:
    """Test full validation with error messages."""
    v = ManifestValidator(sample_manifest)
    errors = v.validate("rsi", {"period": 14}, ["value"])
    assert errors == []
    errors = v.validate("rsi", {"period": 200}, ["value"])
    assert len(errors) == 1
    assert "Invalid parameter" in errors[0]
    errors = v.validate("unknown", {}, [])
    assert len(errors) == 1
    assert "Unknown indicator" in errors[0]
    # Multiple errors
    errors = v.validate(
        "rsi", {"period": 200, "source": 123}, ["invalid_attr"]
    )
    assert len(errors) >= 2


@pytest.mark.unit
@pytest.mark.manifest
def test_manifest_to_dict_from_dict(sample_manifest: Manifest) -> None:
    """Test serialization and deserialization of manifest."""
    manifest_dict = sample_manifest.to_dict()
    reconstructed = Manifest.from_dict(manifest_dict)
    assert reconstructed.indicators.keys() == sample_manifest.indicators.keys()
    assert (
        reconstructed.indicators["rsi"].attributes
        == sample_manifest.indicators["rsi"].attributes
    )
    assert (
        reconstructed.indicators["macd"].parameters.keys()
        == sample_manifest.indicators["macd"].parameters.keys()
    )
