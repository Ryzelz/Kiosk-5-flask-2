from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


ROOT_TEST_FILE = Path(__file__).resolve().parent.parent / "test_app.py"
SPEC = spec_from_file_location("wideye_root_test_app", ROOT_TEST_FILE)
MODULE = module_from_spec(SPEC)

assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

for name in dir(MODULE):
    if name.startswith("_"):
        continue
    globals()[name] = getattr(MODULE, name)
