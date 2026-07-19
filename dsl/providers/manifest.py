"""Structure of the manifest and validator for indicator providers."""

from dataclasses import dataclass, field
from typing import Any

__all__ = (
    'ParameterSchema',
    'IndicatorSchema',
    'Manifest',
    'ManifestValidator'
)


@dataclass(frozen=True, slots=True)
class ParameterSchema:
    """Schema for a single indicator parameter.

    Attributes:
        type: Parameter type ('integer', 'float', 'any').
        default: Default value if not provided.
        min: Minimum allowed value (for numeric types).
        max: Maximum allowed value (for numeric types).

    """

    type: str = 'any'
    default: Any | None = None
    min: float | None = None
    max: float | None = None


@dataclass(frozen=True, slots=True)
class IndicatorSchema:
    """Schema for a single indicator.

    Attributes:
        parameters: Mapping of parameter names to ParameterSchema.
        attributes: List of valid attribute names for this indicator.

    """

    parameters: dict[str, ParameterSchema] = field(default_factory=dict)
    attributes: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Manifest:
    """Provider manifest describing available indicators.

    Attributes:
        indicators: Mapping of indicator names to IndicatorSchema.

    """

    indicators: dict[str, IndicatorSchema] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the manifest to a JSON-serializable dictionary.

        Returns:
            A dict with 'indicators' key containing full schemas.

        """
        result: dict[str, Any] = {'indicators': {}}
        for name, schema in self.indicators.items():
            result['indicators'][name] = {
                'parameters': {},
                'attributes': schema.attributes,
            }
            for param_name, param_schema in schema.parameters.items():
                param_dict: dict[str, Any] = {'type': param_schema.type}
                if param_schema.default is not None:
                    param_dict['default'] = param_schema.default
                if param_schema.min is not None:
                    param_dict['min'] = param_schema.min
                if param_schema.max is not None:
                    param_dict['max'] = param_schema.max
                result['indicators'][name]['parameters'][param_name] = param_dict  # noqa: E501
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Manifest':
        """Reconstruct a Manifest from a dictionary.

        Args:
            data: Dictionary containing 'indicators' key.

        Returns:
            Manifest instance.

        """
        indicators: dict[str, IndicatorSchema] = {}
        for name, schema_data in data.get('indicators', {}).items():
            parameters: dict[str, ParameterSchema] = {
                param_name: ParameterSchema(
                    type=param_data.get('type', 'any'),
                    default=param_data.get('default'),
                    min=param_data.get('min'),
                    max=param_data.get('max'),
                )
                for param_name, param_data in schema_data.get(
                    'parameters', {}
                ).items()
            }
            indicators[name] = IndicatorSchema(
                parameters=parameters,
                attributes=schema_data.get('attributes', []),
            )
        return cls(indicators=indicators)


class ManifestValidator:
    """Validator for indicator requests against a manifest.

    Provides methods to check indicator existence, parameter validity,
    and attribute existence.
    """

    def __init__(
        self,
        manifest: Manifest,
        allow_undefined: bool = False
    ) -> None:
        """Initialize with a manifest.

        Args:
            manifest: The Manifest to validate against.
            allow_undefined: If True, unknown parameters are allowed
            (pass validation). Default is False (strict).

        """
        self.manifest = manifest
        self.allow_undefined = allow_undefined

    def validate_indicator(self, indicator: str) -> bool:
        """Check if the indicator exists in the manifest."""
        return indicator in self.manifest.indicators

    def validate_attribute(self, indicator: str, attribute: str) -> bool:
        """Check if a single attribute exists for the indicator."""
        if indicator not in self.manifest.indicators:
            return False
        schema = self.manifest.indicators[indicator]
        return attribute in schema.attributes

    def validate_attributes(
        self,
        indicator: str,
        attributes: list[str]
    ) -> bool:
        """Check if all given attributes exist for the indicator.

        Args:
            indicator: Name of the indicator.
            attributes: List of attribute names.

        Returns:
            True if all attributes are valid, False otherwise.

        """
        if not attributes:
            return True
        schema = self.manifest.indicators.get(indicator)
        if not schema:
            return False
        valid_attrs = schema.attributes
        return all(attr in valid_attrs for attr in attributes)

    def validate_parameter(
        self,
        indicator: str,
        param_name: str,
        value: Any
    ) -> bool:
        """Validate a single parameter against the manifest schema.

        For numeric parameters (integer/float), range constraints are applied.
        For 'any' type, only type checks are performed (no range constraints).

        Unknown parameters are rejected unless allow_undefined is True.
        """
        if indicator not in self.manifest.indicators:
            return False
        schema = self.manifest.indicators[indicator]
        param_schema = schema.parameters.get(param_name)
        if not param_schema:
            # Strict by default: unknown parameters are not allowed.
            return self.allow_undefined
        # Type checks
        if param_schema.type == 'integer' and not isinstance(value, int):
            return False
        if param_schema.type == 'float' and not isinstance(
            value, (int, float)
        ):
            return False
        # Range checks only for numeric values when constraints are defined
        if isinstance(value, (int, float)):
            if param_schema.min is not None and value < param_schema.min:
                return False
            if param_schema.max is not None and value > param_schema.max:
                return False
        return True

    def validate(
        self,
        indicator: str,
        params: dict[str, Any],
        attributes: list[str],
    ) -> list[str]:
        """Perform full validation of an indicator request.

        Args:
            indicator: Name of the indicator.
            params: Parameter dictionary.
            attributes: List of attribute names.

        Returns:
            List of error messages (empty if all valid).

        """
        errors: list[str] = []
        if not self.validate_indicator(indicator):
            errors.append(f'Unknown indicator: {indicator}')
            return errors
        # schema variable is kept for future extensibility
        schema = self.manifest.indicators[indicator]  # noqa: F841
        errors.extend(
            f"Invalid parameter '{param_name}' for indicator '{indicator}'"
            for param_name, value in params.items()
            if not self.validate_parameter(indicator, param_name, value)
        )
        if not self.validate_attributes(indicator, attributes):
            errors.append(
                f'Invalid attribute path for {indicator}: {attributes}'
            )
        return errors
