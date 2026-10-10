import pytest

from training_framework.hpc.config.requirements import RequirementsError, parse_requirements

SHA = "a" * 40


def test_formatting_does_not_change_the_normalized_list_or_key():
    tidy = parse_requirements("numpy==2.4.4\ntorch==2.13.0\n", "r.txt")
    messy = parse_requirements(
        "# pinned\n\nTorch == 2.13.0   # comment\n"
        "numpy==\\\n2.4.4\n",
        "r.txt",
    )
    assert tidy.normalized() == messy.normalized()
    assert tidy.key() == messy.key()


def test_hashes_continuations_fragments_and_index_options_parse():
    text = (
        "--index-url https://download.pytorch.org/whl/cu128\n"
        "--extra-index-url=https://pypi.org/simple\n"
        "torch==2.13.0 \\\n"
        "    --hash=sha256:ABCDEF \\\n"
        "    --hash=sha256:012345\n"
        f"lib @ git+https://example.org/lib.git@{SHA}#egg=lib\n"
        "data @ https://example.org/data-1.0.tar.gz#sha256=" + "f" * 64 + "\n"
    )
    parsed = parse_requirements(text, "r.txt")
    assert parsed.options == (
        ("--index-url", "https://download.pytorch.org/whl/cu128"),
        ("--extra-index-url", "https://pypi.org/simple"),
    )
    torch = parsed.pins_named("torch")[0]
    assert torch.version == "2.13.0"
    assert torch.hashes == ("sha256:012345", "sha256:abcdef")
    assert parsed.pins_named("lib")[0].url.endswith(f"@{SHA}#egg=lib")
    assert parse_requirements(text, "r.txt").key() == parsed.key()


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("-r other.txt", "option '-r'"),
        ("-c constraints.txt", "option '-c'"),
        ("-e .", "option '-e'"),
        ("--trusted-host example.org", "option '--trusted-host'"),
        ("torch==${TORCH}", "environment-variable"),
        ("torch>=2.0", "not pinned"),
        ("torch==2.*", "not pinned"),
        ("torch", "not pinned"),
        ("lib @ git+https://example.org/lib.git@main", "without a full commit SHA"),
        ("torch==2.13.0 --no-binary :all:", "'--no-binary' is not allowed"),
        ("not a requirement!!", "not a valid requirement"),
    ],
)
def test_refused_forms_name_the_line(line, message):
    with pytest.raises(RequirementsError) as error:
        parse_requirements(f"numpy==2.4.4\n{line}\n", "r.txt")
    assert "r.txt:2" in str(error.value)
    assert message in str(error.value)


def test_a_package_pinned_twice_is_refused():
    with pytest.raises(RequirementsError, match="already pinned on line 1"):
        parse_requirements("torch==2.13.0\nTorch==2.13.0\n", "r.txt")
