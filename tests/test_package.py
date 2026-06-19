import mashup


def test_version_exists():
    assert hasattr(mashup, "__version__")
