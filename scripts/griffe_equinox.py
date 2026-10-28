"""Griffe extension adding the `__init__` that Equinox generates to Equinox modules.

Griffe synthesizes `__init__` for dataclasses only. Equinox modules are dataclasses
at run time, so their API reference needs the same constructor: its signature and
the parameters that `Args:` sections in class docstrings describe. The
documentation build loads this extension through `zensical.toml`.
"""

from typing import Any, cast

import griffe


def _is_equinox_module(cls: griffe.Class) -> bool:
    """Returns whether ``cls`` derives from ``equinox.Module``."""
    if any(
        isinstance(base, griffe.Expr) and base.canonical_path == "equinox.Module"
        for base in cls.bases
    ):
        return True
    return any(
        _is_equinox_module(cast(griffe.Class, base)) for base in cls.resolved_bases if base.is_class
    )


def _field_arguments(attribute: griffe.Attribute) -> dict[str, Any] | None:
    """Returns the keyword arguments of an ``eqx.field(...)`` value, or ``None``."""
    value = attribute.value
    if not isinstance(value, griffe.ExprCall) or value.canonical_path != "equinox.field":
        return None
    return {
        argument.name: argument.value
        for argument in value.arguments
        if isinstance(argument, griffe.ExprKeyword)
    }


def _is_field(attribute: griffe.Attribute) -> bool:
    """Returns whether an annotated class attribute is a dataclass field."""
    annotation = attribute.annotation
    if annotation is None or "property" in attribute.labels:
        return False
    if isinstance(annotation, griffe.ExprSubscript):
        annotation = annotation.left
    return not (
        isinstance(annotation, griffe.Expr)
        and annotation.canonical_path
        in {"equinox.AbstractVar", "equinox.AbstractClassVar", "typing.ClassVar"}
    )


def _parameters(cls: griffe.Class) -> list[griffe.Parameter]:
    """Returns the constructor parameters of the fields of ``cls`` and its parents."""
    try:
        lineage = [*reversed(cls.mro()), cls]
    except ValueError:
        lineage = [cls]
    fields: dict[str, griffe.Parameter] = {}
    for owner in lineage:
        for member in owner.members.values():
            if member.is_alias or not member.is_attribute:
                continue
            attribute = cast(griffe.Attribute, member)
            if not _is_field(attribute):
                continue
            arguments = _field_arguments(attribute)
            if arguments is not None and str(arguments.get("init")) == "False":
                fields.pop(attribute.name, None)
                continue
            if arguments is None:
                default = attribute.value
            elif "default_factory" in arguments:
                default = griffe.ExprCall(function=arguments["default_factory"], arguments=[])
            else:
                default = arguments.get("default")
            keyword_only = arguments is not None and str(arguments.get("kw_only")) == "True"
            fields[attribute.name] = griffe.Parameter(
                attribute.name,
                annotation=attribute.annotation,
                kind=griffe.ParameterKind.keyword_only
                if keyword_only
                else griffe.ParameterKind.positional_or_keyword,
                default=default,
            )
    parameters = list(fields.values())
    return [
        *(p for p in parameters if p.kind is not griffe.ParameterKind.keyword_only),
        *(p for p in parameters if p.kind is griffe.ParameterKind.keyword_only),
    ]


def _classes(obj: griffe.Module | griffe.Class, found: dict[str, griffe.Class]) -> None:
    """Collects the classes defined under ``obj`` into ``found``, keyed by path."""
    for member in obj.members.values():
        if member.is_alias or member.path in found:
            continue
        if member.is_class:
            found[member.path] = cast(griffe.Class, member)
        if member.is_module or member.is_class:
            _classes(cast(griffe.Module | griffe.Class, member), found)


def _add_constructors(package: griffe.Module) -> None:
    """Adds the generated ``__init__`` to every Equinox module of ``package``.

    Equinox modules without their own or an inherited explicit ``__init__`` receive
    one; all are selected before any is added, so added constructors are not
    inherited.
    """
    found: dict[str, griffe.Class] = {}
    _classes(package, found)
    generated = [
        cls
        for cls in found.values()
        if _is_equinox_module(cls) and "__init__" not in cls.all_members
    ]
    for cls in generated:
        cls.set_member(
            "__init__",
            griffe.Function(
                "__init__",
                lineno=0,
                endlineno=0,
                parent=cls,
                parameters=griffe.Parameters(
                    griffe.Parameter("self", kind=griffe.ParameterKind.positional_or_keyword),
                    *_parameters(cls),
                ),
                returns="None",
            ),
        )


class EquinoxModules(griffe.Extension):
    """Griffe extension adding the generated ``__init__`` to Equinox modules."""

    def on_package(self, *, pkg: griffe.Module, **kwargs: Any) -> None:
        """Adds constructors to the Equinox modules of a loaded package.

        Args:
            pkg: Loaded package.
            **kwargs: Other hook arguments.
        """
        _add_constructors(pkg)
