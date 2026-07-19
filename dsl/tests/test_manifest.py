import pytest
from dsl.providers.manifest import (
    Manifest, ManifestValidator, ParameterSchema, IndicatorSchema
)

def make_manifest():
    return Manifest(
        indicators={
            "rsi": IndicatorSchema(
                parameters={
                    "period": ParameterSchema(type="integer", default=14, min=1, max=100),
                    "source": ParameterSchema(type="any", default="close")
                },
                attributes=["value", "signal"]
            ),
            "macd": IndicatorSchema(
                parameters={
                    "fast": ParameterSchema(type="integer", default=12, min=2),
                    "slow": ParameterSchema(type="integer", default=26, min=2),
                },
                attributes=["line", "signal", "histogram"]
            )
        }
    )

def test_valid_indicator():
    v = ManifestValidator(make_manifest())
    assert v.validate_indicator("rsi") == True
    assert v.validate_indicator("unknown") == False

def test_valid_parameters():
    v = ManifestValidator(make_manifest())
    assert v.validate_parameter("rsi", "period", 14) == True
    assert v.validate_parameter("rsi", "period", 200) == False  # > max
    assert v.validate_parameter("rsi", "period", 0) == False    # < min
    assert v.validate_parameter("rsi", "period", 3.14) == False # не integer

def test_undefined_parameter_strict():
    v = ManifestValidator(make_manifest(), allow_undefined=False)
    assert v.validate_parameter("rsi", "unknown_param", 10) == False

def test_undefined_parameter_allowed():
    v = ManifestValidator(make_manifest(), allow_undefined=True)
    assert v.validate_parameter("rsi", "unknown_param", 10) == True

def test_valid_attributes():
    v = ManifestValidator(make_manifest())
    assert v.validate_attributes("rsi", ["value"]) == True
    assert v.validate_attributes("rsi", ["value", "signal"]) == True
    assert v.validate_attributes("rsi", ["nonexistent"]) == False
    assert v.validate_attributes("unknown", []) == True  # пустой список атрибутов всегда True

def test_full_validation():
    v = ManifestValidator(make_manifest())
    errors = v.validate("rsi", {"period": 14}, ["value"])
    assert errors == []
    errors = v.validate("rsi", {"period": 200}, ["value"])
    assert len(errors) == 1
    assert "Invalid parameter" in errors[0]
    errors = v.validate("unknown", {}, [])
    assert len(errors) == 1
    assert "Unknown indicator" in errors[0]