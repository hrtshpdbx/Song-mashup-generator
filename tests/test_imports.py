import importlib


def test_imports():
    modules = [
        "mashup",
        "mashup.cli",
        "mashup.download",
        "mashup.separation",
        "mashup.structure",
        "mashup.features",
        "mashup.scoring",
        "mashup.assembly",
        "mashup.evaluation",
    ]

    for module in modules:
        assert importlib.import_module(module) is not None
