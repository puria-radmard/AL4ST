from . import acquisition
from . import agent
from . import batch_querying
from . import beam_search
from . import selector
from . import util_classes


def disable_tqdm():
    for mdl in [
        acquisition,
        agent,
        batch_querying,
        beam_search,
        selector,
        util_classes,
    ]:
        mdl.TQDM_MODE = False
