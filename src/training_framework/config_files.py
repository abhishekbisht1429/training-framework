"""Loading a configuration file whose sessions include other files.

A session entry may list `includes`: files holding fragments of a session
(a model, an optimizer recipe, data settings). They are merged in order,
then the entry itself on top; a fragment may include others. Paths are
relative to the file that names them.

```yaml
sessions:
  - includes: [common/vit_s16.yaml, common/dino_optimizer.yaml]
    session_config: {...}
    optimizer: {optimizer: {kwargs: {lr: 0.0005}}}
```

Merging is strict, so a setting cannot change without anyone seeing it:
mappings merge key by key, every other value (lists included) is replaced
whole; two fragments setting one value differently is an error naming both
files -- only the file that includes them may override. OmegaConf
interpolation (`${...}`) is refused: values are never resolved, so it would
reach a component as a literal string. `\\${` writes a literal `${`.

The result has no `includes` left: a run records one complete config.
`ConfigSource`s say which files each session was built from.
"""

from __future__ import annotations

import hashlib
import os
import posixpath
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import yaml
from omegaconf import OmegaConf

INCLUDES_KEY = "includes"
#: Written next to a run's `config.yaml`: the files its config came from.
SOURCES_FILE = "config_sources.yaml"

# `${` not preceded by a backslash; `\${` is the escape for a literal `${`.
_INTERPOLATION = re.compile(r"(?<!\\)\$\{")
_ESCAPED = re.compile(r"\\\$\{")


class ConfigFileError(ValueError):
    """A configuration file, or a file it includes, cannot be used."""


@dataclass(frozen=True)
class ConfigSource:
    """A file a session's config was built from."""

    path: str
    sha256: str
    #: The file that included it; None for the file that was loaded.
    included_by: str | None

    def as_json(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "included_by": self.included_by}


@dataclass
class LoadedConfig:
    """A whole configuration file, includes merged.

    `sources[i]` lists the files session `i` was built from: the file
    loaded, then each included file in the order it was read.
    """

    config: dict[str, Any]
    sources: list[list[ConfigSource]]


class ConfigReader(Protocol):
    def read(self, path: str) -> bytes | None:
        """The file's bytes, or None when there is no such file."""

    def resolve(self, path: str, relative_to: str | None) -> str:
        """`path` as named in the file `relative_to` (None: the file loaded),
        as the key the file is read and reported by."""


class LocalFiles:
    """Files on this machine; an included path may be absolute."""

    def read(self, path: str) -> bytes | None:
        try:
            with open(path, "rb") as handle:
                return handle.read()
        except FileNotFoundError:
            return None
        except OSError as error:
            raise ConfigFileError(f"{path}: cannot be read ({error})") from None

    def resolve(self, path: str, relative_to: str | None) -> str:
        if relative_to is not None and not os.path.isabs(path):
            path = os.path.join(os.path.dirname(relative_to), path)
        return os.path.realpath(path)


class RepositoryFiles:
    """Files of a repository at one commit, by repository-relative path; an
    included path may not leave the repository, so the commit is the whole
    source of the config."""

    def __init__(self, read) -> None:
        self._read = read

    def read(self, path: str) -> bytes | None:
        return self._read(path)

    def resolve(self, path: str, relative_to: str | None) -> str:
        if posixpath.isabs(path) or "\\" in path:
            raise ConfigFileError(
                f"include {path!r}: a batch's config can only include files of "
                "the repository, by a relative path with / separators"
            )
        base = posixpath.dirname(relative_to) if relative_to is not None else ""
        resolved = posixpath.normpath(posixpath.join(base, path))
        if resolved == ".." or resolved.startswith("../"):
            raise ConfigFileError(
                f"include {path!r} (in {relative_to}) leaves the repository; a "
                "batch's config can only include files of the repository"
            )
        return resolved


def load_config(path: str | os.PathLike, *, reader: ConfigReader | None = None) -> LoadedConfig:
    """The configuration file at `path` with every session's includes
    merged. Raises `ConfigFileError` naming the file (and the chain of
    includes) for anything that cannot be used.

    Escaped `\\${` stay escaped, so the result can be written out and loaded
    again; `unescape` it last, where the values are used.
    """
    reader = reader or LocalFiles()
    top = reader.resolve(os.fspath(path), None)
    config, digest = _read_mapping(reader, top, (), what="configuration")
    if INCLUDES_KEY in config:
        raise ConfigFileError(
            f"{top}: {INCLUDES_KEY!r} belongs in a session entry "
            f"(sessions[i].{INCLUDES_KEY}), not at the top of the file"
        )
    sessions = config.get("sessions")
    all_sources: list[list[ConfigSource]] = []
    if isinstance(sessions, list):
        merged_sessions = []
        for index, entry in enumerate(sessions):
            sources = [ConfigSource(top, digest, None)]
            if isinstance(entry, Mapping):
                entry = _expand(reader, dict(entry), top, (top,), sources, f"sessions[{index}]")
            merged_sessions.append(entry)
            all_sources.append(sources)
        config = {**config, "sessions": merged_sessions}
    return LoadedConfig(config=config, sources=all_sources)


def check_no_interpolation(value: Any, where: str) -> None:
    """Refuse `${...}`: the config is never resolved, so an interpolation
    would reach a component as the literal text."""
    for path, text in _strings(value, ()):
        if _INTERPOLATION.search(text):
            location = ".".join(str(part) for part in path) or "(value)"
            raise ConfigFileError(
                f"{where}: {location} is {text!r}; interpolation (${{...}}) is not "
                "supported -- values are not resolved, so it would reach the "
                "component as this text. Share settings with includes; write "
                "\\${ for a literal ${"
            )


def unescape(value: Any) -> Any:
    """`\\${` back to a literal `${`, everywhere in `value`."""
    if isinstance(value, str):
        return _ESCAPED.sub("${", value)
    if isinstance(value, Mapping):
        return {key: unescape(item) for key, item in value.items()}
    if isinstance(value, list):
        return [unescape(item) for item in value]
    return value


def sources_document(sources: Sequence[ConfigSource]) -> dict[str, Any]:
    """What `config_sources.yaml` holds."""
    return {"sources": [source.as_json() for source in sources]}


def write_config_sources(session_dir: str, sources: Sequence[ConfigSource]) -> str:
    """Write `config_sources.yaml` into a run directory; its path."""
    os.makedirs(session_dir, exist_ok=True)
    path = os.path.join(session_dir, SOURCES_FILE)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(sources_document(sources), handle, sort_keys=False)
    return path


# -- reading and merging -------------------------------------------------------------


def _read_mapping(reader, path, chain, *, what) -> tuple[dict, str]:
    raw = reader.read(path)
    if raw is None:
        raise ConfigFileError(_with_chain(f"{path}: no such file", chain))
    try:
        loaded = OmegaConf.to_container(OmegaConf.create(raw.decode("utf-8")))
    except Exception as error:  # OmegaConf and YAML raise several types
        raise ConfigFileError(_with_chain(f"{path}: not readable YAML ({error})", chain)) from None
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise ConfigFileError(_with_chain(
            f"{path}: a {what} file must be a mapping, not a {type(loaded).__name__}", chain,
        ))
    check_no_interpolation(loaded, _with_chain(path, chain))
    return loaded, hashlib.sha256(raw).hexdigest()


def _expand(reader, entry, file, chain, sources, where) -> dict:
    """`entry` with its includes merged under it."""
    includes = entry.pop(INCLUDES_KEY, None)
    if includes is None:
        return entry
    if (
            not isinstance(includes, list)
            or not all(isinstance(item, str) and item for item in includes)
    ):
        raise ConfigFileError(_with_chain(
            f"{file}: {where}.{INCLUDES_KEY} must be a list of file paths; got {includes!r}",
            chain[:-1],
        ))
    layers: list[tuple[str, dict]] = []
    for name in includes:
        path = reader.resolve(name, file)
        if path in chain:
            cycle = " -> ".join((*chain[chain.index(path):], path))
            raise ConfigFileError(f"includes form a cycle: {cycle}")
        fragment, digest = _read_mapping(reader, path, chain, what="fragment")
        if "sessions" in fragment:
            raise ConfigFileError(_with_chain(
                f"{path}: an included file is a fragment of a session entry and "
                "cannot have 'sessions'", chain,
            ))
        sources.append(ConfigSource(path, digest, file))
        layers.append((path, _expand(reader, fragment, path, (*chain, path), sources, "(fragment)")))
    merged = _merge_fragments(layers, file)
    return _overlay(merged, entry)


def _merge_fragments(layers: list[tuple[str, dict]], file: str) -> dict:
    """Fragments merged in order; one fragment changing a value another set
    is an error, so their order never decides a setting."""
    merged: dict = {}
    origin: dict[tuple, str] = {}
    for path, layer in layers:
        for leaf, value in _leaves(layer, ()):
            for depth in range(1, len(leaf) + 1):
                prefix = leaf[:depth]
                if prefix not in origin:
                    continue
                if depth < len(leaf):
                    earlier = _get(merged, prefix)
                    if earlier is None or isinstance(earlier, dict):
                        continue  # a null entry, the same as an empty one
                    clash = f"{_dotted(prefix)} is a value in {origin[prefix]} and a mapping in {path}"
                else:
                    earlier = _get(merged, prefix)
                    if earlier == value and type(earlier) is type(value):
                        break
                    clash = (
                        f"{_dotted(prefix)} is {earlier!r} in {origin[prefix]} and "
                        f"{value!r} in {path}"
                    )
                raise ConfigFileError(
                    f"{file}: two included files set the same value: {clash}. "
                    "Set it in one of them, or override it in the including file"
                )
            mapping_prefix = _mapping_leaf_under(origin, leaf)
            if mapping_prefix is not None and value is not None:
                raise ConfigFileError(
                    f"{file}: two included files set the same value: "
                    f"{_dotted(leaf)} is a mapping in {origin[mapping_prefix]} and "
                    f"{value!r} in {path}. Set it in one of them, or override it "
                    "in the including file"
                )
            origin[leaf] = path
        merged = _overlay(merged, layer)
    return merged


def _mapping_leaf_under(origin, leaf):
    """An earlier leaf strictly below `leaf` (so `leaf` was a mapping)."""
    for known in origin:
        if len(known) > len(leaf) and known[:len(leaf)] == leaf:
            return known
    return None


def _overlay(base: Any, top: Any) -> Any:
    """`top` merged onto `base`: mappings key by key, anything else
    replaced; a null merged with a mapping is that mapping."""
    if isinstance(base, dict) and isinstance(top, dict):
        result = dict(base)
        for key, value in top.items():
            result[key] = _overlay(base[key], value) if key in base else value
        return result
    if top is None and isinstance(base, dict):
        return base
    return top


def _leaves(value: Any, path: tuple):
    """(path, value) of every value that is not a mapping; an empty
    mapping has none."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _leaves(item, (*path, key))
    else:
        yield path, value


def _get(config: dict, path: tuple) -> Any:
    value: Any = config
    for key in path:
        value = value.get(key) if isinstance(value, dict) else None
    return value


def _strings(value: Any, path: tuple):
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _strings(item, (*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _strings(item, (*path, f"[{index}]"))


def _dotted(path: tuple) -> str:
    return ".".join(str(part) for part in path)


def _with_chain(message: str, chain: tuple) -> str:
    if len(chain) <= 1:
        return message
    return f"{message} (included via {' -> '.join(chain)})"
