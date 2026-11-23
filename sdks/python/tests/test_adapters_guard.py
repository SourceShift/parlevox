"""Static guard: no ``raise`` inside an ``except`` handler without ``from None``.

A ``raise`` executed inside an ``except`` handler implicitly records the
in-flight exception as the new exception's ``__context__``. A raw transport
error may carry request material (a credential in a URL or header value), so a
classified :class:`~parlevox_sdk.adapters.base.AdapterError` raised there would
render that material through ``traceback.format_exc()`` even though its own
message is redacted. Every in-handler raise must therefore be either ``raise
... from None`` or one of the intentional bare re-raises allow-listed below.
"""

from __future__ import annotations

import ast
import dataclasses
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ADAPTERS_DIR = REPO_ROOT / "sdks" / "python" / "parlevox_sdk" / "adapters"

#: ``(relative file, dotted enclosing function, normalized raise source)``.
#: Each entry is a deliberate bare re-raise that must preserve the original
#: exception type and identity for the caller to classify.
ALLOWED_IN_HANDLER_RAISES = {
    # ``except AdapterError: raise`` passes an already-classified provider
    # failure through untouched; wrapping or chaining it would lose identity.
    ("sdks/python/parlevox_sdk/adapters/base.py", "Transport.send", "raise"): (
        "bare re-raise of an already-classified AdapterError"
    ),
    # The body iterator re-raises an AdapterError without re-wrapping it.
    (
        "sdks/python/parlevox_sdk/adapters/base.py",
        "TransportResponse.iter_bytes",
        "raise",
    ): "bare re-raise of an already-classified AdapterError",
    # ``Transport.send`` classifies by exception type, so this re-raise must
    # keep the original OSError/HTTPException rather than being chained away.
    (
        "sdks/python/parlevox_sdk/adapters/base.py",
        "HttpTransport._send_once",
        "raise",
    ): "bare re-raise of the raw transport error for outer classification",
}


@dataclasses.dataclass(frozen=True)
class InHandlerRaise:
    """One offending ``raise`` statement inside an ``except`` handler."""

    filename: str
    lineno: int
    function: str
    handler: str
    raise_source: str

    def __str__(self):
        return (
            f"{self.filename}:{self.lineno} in {self.function or '<module>'}: "
            f"{self.raise_source} inside {self.handler}"
        )


def _handler_description(handler):
    type_source = ast.unparse(handler.type) if handler.type is not None else ""
    return f"{type_source or ':'} handler at line {handler.lineno}"


def find_in_handler_raises(tree, filename):
    """Return every raise executed directly inside an ``except`` handler.

    The walker propagates the in-handler flag into nested ``try`` bodies (a
    raise there still executes inside the handler) but resets it for
    ``FunctionDef``/``AsyncFunctionDef``/``Lambda`` bodies, which are separate
    execution scopes. Class bodies keep the enclosing flag: a class statement
    executes in place.
    """
    violations = []
    _visit(
        tree,
        in_handler=False,
        scope=(),
        handler=None,
        filename=filename,
        violations=violations,
    )
    return violations


def _visit(node, *, in_handler, scope, handler, filename, violations):
    if isinstance(node, ast.Raise):
        _check_raise(
            node,
            in_handler=in_handler,
            scope=scope,
            handler=handler,
            filename=filename,
            violations=violations,
        )
        for child in (node.exc, node.cause):
            if child is not None:
                _visit(
                    child,
                    in_handler=in_handler,
                    scope=scope,
                    handler=handler,
                    filename=filename,
                    violations=violations,
                )
        return

    if isinstance(node, ast.ClassDef):
        inner = scope + (node.name,)
        for decorator in node.decorator_list:
            _visit(
                decorator,
                in_handler=in_handler,
                scope=scope,
                handler=handler,
                filename=filename,
                violations=violations,
            )
        for stmt in node.body:
            _visit(
                stmt,
                in_handler=in_handler,
                scope=inner,
                handler=handler,
                filename=filename,
                violations=violations,
            )
        return

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        inner = scope + (node.name,)
        for decorator in node.decorator_list:
            _visit(
                decorator,
                in_handler=in_handler,
                scope=scope,
                handler=handler,
                filename=filename,
                violations=violations,
            )
        for default in node.args.defaults + [
            default for default in node.args.kw_defaults if default is not None
        ]:
            _visit(
                default,
                in_handler=in_handler,
                scope=scope,
                handler=handler,
                filename=filename,
                violations=violations,
            )
        for stmt in node.body:
            _visit(
                stmt,
                in_handler=False,
                scope=inner,
                handler=None,
                filename=filename,
                violations=violations,
            )
        return

    if isinstance(node, ast.Lambda):
        for default in node.args.defaults + [
            default for default in node.args.kw_defaults if default is not None
        ]:
            _visit(
                default,
                in_handler=in_handler,
                scope=scope,
                handler=handler,
                filename=filename,
                violations=violations,
            )
        _visit(
            node.body,
            in_handler=False,
            scope=scope + ("<lambda>",),
            handler=None,
            filename=filename,
            violations=violations,
        )
        return

    try_nodes = (ast.Try, ast.TryStar) if hasattr(ast, "TryStar") else (ast.Try,)
    if isinstance(node, try_nodes):
        for stmt in node.body:
            _visit(
                stmt,
                in_handler=in_handler,
                scope=scope,
                handler=handler,
                filename=filename,
                violations=violations,
            )
        for clause in node.handlers:
            if clause.type is not None:
                _visit(
                    clause.type,
                    in_handler=in_handler,
                    scope=scope,
                    handler=handler,
                    filename=filename,
                    violations=violations,
                )
            description = _handler_description(clause)
            for stmt in clause.body:
                _visit(
                    stmt,
                    in_handler=True,
                    scope=scope,
                    handler=description,
                    filename=filename,
                    violations=violations,
                )
        for stmt in node.orelse + node.finalbody:
            _visit(
                stmt,
                in_handler=in_handler,
                scope=scope,
                handler=handler,
                filename=filename,
                violations=violations,
            )
        return

    for child in ast.iter_child_nodes(node):
        _visit(
            child,
            in_handler=in_handler,
            scope=scope,
            handler=handler,
            filename=filename,
            violations=violations,
        )


def _check_raise(node, *, in_handler, scope, handler, filename, violations):
    if not in_handler:
        return
    if isinstance(node.cause, ast.Constant) and node.cause.value is None:
        return
    function = ".".join(scope)
    source = ast.unparse(node).strip()
    if (filename, function, source) in ALLOWED_IN_HANDLER_RAISES:
        return
    violations.append(
        InHandlerRaise(
            filename=filename,
            lineno=node.lineno,
            function=function,
            handler=handler or "<unknown handler>",
            raise_source=source,
        )
    )


def _scan_adapters():
    violations = []
    for path in sorted(ADAPTERS_DIR.glob("*.py")):
        display = path.relative_to(REPO_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=display)
        violations.extend(find_in_handler_raises(tree, display))
    return violations


def test_no_raise_inside_except_handlers():
    violations = _scan_adapters()
    assert not violations, "in-handler raise without `from None`:\n" + "\n".join(
        str(violation) for violation in violations
    )


def test_detector_flags_a_planted_in_handler_raise():
    """Self-test: the detector rejects a raise that is not ``from None``."""
    source = textwrap.dedent(
        """
        def handler_owner():
            try:
                work()
            except ValueError:
                raise RuntimeError("boom")
        """
    )
    violations = find_in_handler_raises(ast.parse(source), "planted.py")
    assert len(violations) == 1
    assert violations[0].function == "handler_owner"
    assert "RuntimeError" in violations[0].raise_source
    assert "ValueError" in violations[0].handler


def test_detector_accepts_from_none_and_skips_nested_scopes():
    """Self-test: ``from None`` passes and nested function scopes are skipped."""
    source = textwrap.dedent(
        """
        def from_none():
            try:
                work()
            except ValueError:
                raise RuntimeError("boom") from None


        def nested_scope():
            try:
                work()
            except ValueError:
                def later():
                    raise RuntimeError("deferred")
                return later
        """
    )
    assert find_in_handler_raises(ast.parse(source), "planted.py") == []


def test_detector_propagates_in_handler_into_nested_try():
    """A raise in a nested try body still executes inside the outer handler."""
    source = textwrap.dedent(
        """
        def outer():
            try:
                work()
            except ValueError:
                try:
                    cleanup()
                except OSError:
                    raise RuntimeError("boom")
        """
    )
    violations = find_in_handler_raises(ast.parse(source), "planted.py")
    assert len(violations) == 1
    assert violations[0].lineno == 9
