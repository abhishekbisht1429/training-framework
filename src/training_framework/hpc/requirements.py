"""The strict pinned-requirements tokenizer.

`packaging` parses PEP 508 requirement text only; the requirements-file
syntax around it (continuations, comments, option lines, `--hash`) is
handled here, strictly, without pip internals. Anything not explicitly
allowed is an error naming the line.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

from training_framework.hpc.names import hash_value

#: Option lines allowed on their own, each with exactly one value.
INDEX_OPTIONS = ("--index-url", "--extra-index-url", "--find-links")

_HASH = re.compile(r"\A--hash=([a-z0-9]+):([0-9a-fA-F]+)\Z")
_FULL_SHA = re.compile(r"@[0-9a-f]{40}(?:[#?]|\Z)")
_ARCHIVE_HASH = re.compile(r"#(?:sha256|sha384|sha512)=[0-9a-fA-F]+")
_ENV_REFERENCE = re.compile(r"\$\{[^}]*\}")


@dataclass(frozen=True)
class Pin:
    name: str           # canonical name
    text: str           # normalized requirement text
    version: str | None  # the `==` version, or None for a direct URL
    url: str | None
    hashes: tuple[str, ...]


@dataclass(frozen=True)
class Requirements:
    pins: tuple[Pin, ...]
    options: tuple[tuple[str, str], ...]

    def normalized(self) -> dict:
        """What the environment key hashes: formatting is not part of it."""
        return {
            "options": [list(option) for option in self.options],
            "pins": [
                {"text": pin.text, "hashes": list(pin.hashes)}
                for pin in self.pins
            ],
        }

    def key(self) -> str:
        return hash_value(self.normalized())

    def pins_named(self, name: str) -> list[Pin]:
        wanted = canonicalize_name(name)
        return [pin for pin in self.pins if pin.name == wanted]


class RequirementsError(ValueError):
    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("\n".join(problems))


def _logical_lines(text: str) -> list[tuple[int, str]]:
    """Join `\\` continuations; return (first line number, content)."""
    lines: list[tuple[int, str]] = []
    pending = ""
    start = 0
    for number, line in enumerate(text.splitlines(), start=1):
        if not pending:
            start = number
        if line.endswith("\\"):
            pending += line[:-1] + " "
            continue
        lines.append((start, pending + line))
        pending = ""
    if pending:
        lines.append((start, pending))
    return lines


def _strip_comment(line: str) -> str:
    # A comment starts at '#' at the beginning of the line or after
    # whitespace; a '#' inside a token (a URL fragment) is kept.
    for index, char in enumerate(line):
        if char == "#" and (index == 0 or line[index - 1].isspace()):
            return line[:index]
    return line


def parse_requirements(text: str, source: str) -> Requirements:
    problems: list[str] = []
    pins: list[Pin] = []
    options: list[tuple[str, str]] = []
    seen: dict[str, int] = {}
    for number, raw in _logical_lines(text):
        where = f"{source}:{number}"
        line = _strip_comment(raw).strip()
        if not line:
            continue
        if _ENV_REFERENCE.search(line):
            problems.append(
                f"{where}: environment-variable references are not allowed"
            )
            continue
        tokens = line.split()
        if tokens[0].startswith("-"):
            option, value = _option(tokens)
            if option not in INDEX_OPTIONS or value is None:
                problems.append(
                    f"{where}: option '{tokens[0]}' is not allowed (allowed: "
                    f"{', '.join(INDEX_OPTIONS)} with one value; "
                    "--hash after a requirement)"
                )
                continue
            options.append((option, value))
            continue
        requirement_tokens = []
        hashes = []
        bad_option = None
        for token in tokens:
            if token.startswith("--"):
                match = _HASH.match(token)
                if match is None:
                    bad_option = token
                    break
                hashes.append(f"{match.group(1)}:{match.group(2).lower()}")
            elif hashes:
                bad_option = token
                break
            else:
                requirement_tokens.append(token)
        if bad_option is not None:
            problems.append(
                f"{where}: '{bad_option}' is not allowed after a requirement "
                "(only --hash=<algorithm>:<hex>)"
            )
            continue
        text_part = " ".join(requirement_tokens)
        try:
            requirement = Requirement(text_part)
        except InvalidRequirement as error:
            problems.append(f"{where}: not a valid requirement ({error})")
            continue
        pin, problem = _pin(requirement, tuple(sorted(hashes)))
        if problem:
            problems.append(f"{where}: {problem}")
            continue
        if pin.name in seen:
            problems.append(
                f"{where}: '{pin.name}' is already pinned on line "
                f"{seen[pin.name]}"
            )
            continue
        seen[pin.name] = number
        pins.append(pin)
    if problems:
        raise RequirementsError(problems)
    pins.sort(key=lambda pin: pin.name)
    return Requirements(tuple(pins), tuple(options))


def _option(tokens: list[str]) -> tuple[str, str | None]:
    first = tokens[0]
    if "=" in first:
        option, value = first.split("=", 1)
        return option, value if len(tokens) == 1 and value else None
    if len(tokens) == 2:
        return first, tokens[1]
    return first, None


def _pin(requirement: Requirement, hashes: tuple[str, ...]):
    name = canonicalize_name(requirement.name)
    if requirement.url is not None:
        url = requirement.url
        if not (_FULL_SHA.search(url) or _ARCHIVE_HASH.search(url)):
            return None, (
                f"'{requirement.name}' is a direct URL without a full commit "
                "SHA or archive hash; pin it"
            )
        return Pin(name, _normalized_text(name, requirement), None, url, hashes), None
    specifiers = list(requirement.specifier)
    if (
            len(specifiers) != 1
            or specifiers[0].operator != "=="
            or specifiers[0].version.endswith(".*")
    ):
        return None, (
            f"'{requirement.name}' is not pinned to one exact version "
            f"(got '{requirement.specifier or 'no version'}'); use ==X.Y.Z"
        )
    return Pin(
        name, _normalized_text(name, requirement), specifiers[0].version,
        None, hashes,
    ), None


def _normalized_text(name: str, requirement: Requirement) -> str:
    """The requirement with its canonical name, so spelling and spacing do
    not change the environment key."""
    extras = f"[{','.join(sorted(requirement.extras))}]" if requirement.extras else ""
    if requirement.url is not None:
        text = f"{name}{extras} @ {requirement.url}"
    else:
        text = f"{name}{extras}{requirement.specifier}"
    if requirement.marker is not None:
        text += f" ; {requirement.marker}"
    return text
