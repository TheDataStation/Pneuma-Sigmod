# We need to import index generator because it is the first import in pneuma.py.
# If we don't have this import in this init file, we will get ModuleNotFoundError.
# when running the code as a pip-installed package.s
# However, for some reason, we only need to import one module and the rest
# will import just fine.
from .index_generator.index_generator import IndexGenerator
from .pneuma import Pneuma
