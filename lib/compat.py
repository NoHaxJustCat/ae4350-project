"""Load models saved before the lib/ reorganisation.

SB3 pickles policy_kwargs, which holds features_extractor_class as a CLASS
REFERENCE -- "libs.policies.SmartEncoder". Renaming the module therefore makes
every existing checkpoint unloadable with ModuleNotFoundError. Aliasing the old
module paths in sys.modules lets pickle resolve them to the new locations, so
out/vbar_specialist and every trained checkpoint keep working.

Import for the side effect only; lib/__init__.py does this on package import.
"""

import sys

import lib.astro.dynamics
import lib.astro.reference
import lib.plots.diagnostics
import lib.plots.style
import lib.plots.trajectory
import lib.rl.env
import lib.rl.net
import lib.rl.obs
import lib.rl.symmetry
import config

_ALIASES = {
    "libs": lib,
    "libs.constants": config,
    "libs.dynamics": lib.astro.dynamics,
    "libs.reference": lib.astro.reference,
    "libs.env": lib.rl.env,
    "libs.normalization": lib.rl.obs,
    "libs.symmetry": lib.rl.symmetry,
    "libs.policies": lib.rl.net,
    "libs.plotting": lib.plots.style,
    "libs.trajectory": lib.plots.trajectory,
    "libs.diagnostics": lib.plots.diagnostics,
}


def install():
    for old, new in _ALIASES.items():
        sys.modules.setdefault(old, new)
