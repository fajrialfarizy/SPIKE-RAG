"""
Offline validator for the Groq Structured Outputs subset the app relies on.

Fix pack 11 (S1c). The rules below are transcribed from
https://console.groq.com/docs/structured-outputs (read 2026-11-22) and nothing
else is enforced:

Documented types
    string, number, boolean, integer, object, array, enum, ``anyOf``, and
    ``null`` (it appears in the documented optional-field examples).

Documented keywords
    ``type``, ``properties``, ``required``, ``additionalProperties``, ``items``,
    ``enum``, ``anyOf``, ``description``, ``minimum``, ``maximum``, ``$ref``,
    ``$defs``.

Documented rules for ``strict: true``
    - every property must be listed in ``required`` (no optional fields),
    - every object must set ``additionalProperties: false``,
    - ``required`` names must exist in ``properties``,
    - an array must declare ``items``,
    - an ``enum`` is a non-empty list of scalars of the declared type,
    - the schema name is a-z, A-Z, 0-9, ``_`` or ``-`` with at most 64 chars.

NOT documented, therefore NOT enforced here (stated rather than invented)
    - nesting depth limits,
    - property-count limits,
    - property-name character rules,
    - ``minItems``/``maxItems`` (the docs never show them; the app caps lists in
      the backend instead),
    - integer bounds (``minimum``/``maximum`` are only shown on ``number`` in the
      docs, so the app uses no bounds at all and validates ranges itself).

``$ref`` nodes are accepted when the target exists in ``$defs`` but are not
traversed: no schema the app sends uses them.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import config

DOCUMENTED_TYPES = frozenset(
    {"string", "number", "boolean", "integer", "object", "array", "null"}
)
DOCUMENTED_KEYWORDS = frozenset(
    {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
        "anyOf",
        "description",
        "minimum",
        "maximum",
        "$ref",
        "$defs",
    }
)
_SCHEMA_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class SchemaRuleError(ValueError):
    """Raised when a schema uses something Groq strict mode does not document."""

    def __init__(self, violations: Sequence[str]):
        self.violations = list(violations)
        super().__init__("; ".join(self.violations))


def _declared_types(node: Mapping[str, Any]) -> List[str]:
    raw = node.get("type")
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(item) for item in raw]
    return []


def _check_enum(node: Mapping[str, Any], path: str, violations: List[str]) -> None:
    values = node.get("enum")
    if values is None:
        return
    if not isinstance(values, list) or not values:
        violations.append(f"{path}: 'enum' must be a non-empty list")
        return
    types = set(_declared_types(node))
    for value in values:
        if isinstance(value, bool):
            ok = "boolean" in types
        elif isinstance(value, int):
            ok = bool(types & {"integer", "number"})
        elif isinstance(value, float):
            ok = "number" in types
        elif isinstance(value, str):
            ok = "string" in types
        else:
            ok = False
        if not ok:
            violations.append(
                f"{path}: enum value {value!r} does not match the declared type"
            )


def _check_node(
    node: Any,
    path: str,
    violations: List[str],
    defs: Mapping[str, Any],
    seen_refs: Iterable[str] = (),
) -> None:
    if not isinstance(node, Mapping):
        violations.append(f"{path}: a schema node must be an object")
        return

    unknown = sorted(set(node) - DOCUMENTED_KEYWORDS)
    if unknown:
        violations.append(
            f"{path}: keyword(s) not documented for strict mode: {', '.join(unknown)}"
        )

    ref = node.get("$ref")
    if ref is not None:
        if not isinstance(ref, str) or not ref.startswith("#/$defs/"):
            violations.append(f"{path}: unsupported $ref {ref!r}")
            return
        name = ref[len("#/$defs/"):]
        if name not in defs:
            violations.append(f"{path}: $ref target {name!r} is not defined")
        return

    types = _declared_types(node)
    if not types:
        violations.append(f"{path}: every node needs a 'type'")
        return
    for declared in types:
        if declared not in DOCUMENTED_TYPES:
            violations.append(f"{path}: type {declared!r} is not documented")

    _check_enum(node, path, violations)

    for key in ("minimum", "maximum"):
        value = node.get(key)
        if value is not None and not isinstance(value, (int, float)):
            violations.append(f"{path}: '{key}' must be a number")

    any_of = node.get("anyOf")
    if any_of is not None:
        if not isinstance(any_of, list) or not any_of:
            violations.append(f"{path}: 'anyOf' must be a non-empty list")
        else:
            for index, item in enumerate(any_of):
                _check_node(item, f"{path}.anyOf[{index}]", violations, defs)

    if "object" in types:
        properties = node.get("properties")
        if not isinstance(properties, Mapping):
            violations.append(f"{path}: an object must declare 'properties'")
            properties = {}
        required = node.get("required")
        if not isinstance(required, list):
            violations.append(f"{path}: every object must declare 'required'")
            required = []
        missing = [name for name in required if name not in properties]
        if missing:
            violations.append(
                f"{path}: required name(s) without a property: {', '.join(missing)}"
            )
        not_required = [name for name in properties if name not in required]
        if not_required:
            violations.append(
                f"{path}: strict mode requires every property, missing: "
                f"{', '.join(not_required)}"
            )
        if node.get("additionalProperties") is not False:
            violations.append(f"{path}: objects must set additionalProperties: false")
        for name, child in properties.items():
            _check_node(child, f"{path}.{name}", violations, defs)

    if "array" in types:
        items = node.get("items")
        if items is None:
            violations.append(f"{path}: an array must declare 'items'")
        else:
            _check_node(items, f"{path}[]", violations, defs)


def check_schema(schema: Mapping[str, Any]) -> List[str]:
    """Every documented violation in one schema; an empty list means valid."""
    violations: List[str] = []
    if not isinstance(schema, Mapping):
        return ["schema: the schema must be a JSON object"]
    defs = schema.get("$defs") or {}
    if not isinstance(defs, Mapping):
        return ["$defs: must be an object"]
    for name, definition in defs.items():
        _check_node(definition, f"$defs.{name}", violations, defs)
    _check_node(schema, "schema", violations, defs)
    return violations


def assert_valid_schema(schema: Mapping[str, Any]) -> None:
    """Raise :class:`SchemaRuleError` when the schema leaves the documented subset."""
    violations = check_schema(schema)
    if violations:
        raise SchemaRuleError(violations)


def check_response_format(response_format: Mapping[str, Any]) -> List[str]:
    """Validate the whole ``response_format`` envelope the client sends."""
    violations: List[str] = []
    if not isinstance(response_format, Mapping):
        return ["response_format: must be an object"]
    if response_format.get("type") != "json_schema":
        violations.append("response_format.type must be 'json_schema'")
    json_schema = response_format.get("json_schema")
    if not isinstance(json_schema, Mapping):
        violations.append("response_format.json_schema must be an object")
        return violations
    if json_schema.get("strict") is not True:
        violations.append("json_schema.strict must be true")
    name = json_schema.get("name")
    if not isinstance(name, str) or not _SCHEMA_NAME.match(name):
        violations.append(
            "json_schema.name must be 1-64 chars of a-z, A-Z, 0-9, '_' or '-'"
        )
    schema = json_schema.get("schema")
    if not isinstance(schema, Mapping):
        violations.append("json_schema.schema must be an object")
        return violations
    violations.extend(check_schema(schema))
    return violations


def assert_valid_response_format(response_format: Mapping[str, Any]) -> None:
    violations = check_response_format(response_format)
    if violations:
        raise SchemaRuleError(violations)


def app_response_formats() -> Dict[str, Mapping[str, Any]]:
    """Every structured-output schema the app sends, by name (offline only)."""
    from core import compare_answer, structured_answer

    return {
        "structured_answer": structured_answer.response_format(),
        "compare_answer": compare_answer.response_format(["g1", "g2"]),
    }


def undocumented_rules() -> List[str]:
    """The rules the docs do not state, so nothing here invents them."""
    return [
        "nesting depth limits",
        "property-count limits",
        "property-name character rules",
        "minItems/maxItems",
        "integer minimum/maximum (documented only on 'number' in the examples)",
    ]


def summarise(schema: Optional[Mapping[str, Any]] = None) -> str:
    """One-line verdict for a script or a test failure message."""
    target = config.ANSWER_JSON_SCHEMA_NAME if schema is None else "schema"
    violations = check_schema(schema) if schema is not None else []
    if not violations:
        return f"{target}: within the documented Groq strict subset"
    return f"{target}: {len(violations)} violation(s): " + "; ".join(violations)
