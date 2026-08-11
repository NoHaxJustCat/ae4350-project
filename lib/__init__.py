"""Project library. Importing installs aliases so models saved before the
lib/ reorganisation still load -- see lib/compat.py."""


def _install_compat():
    from lib import compat
    compat.install()


_install_compat()
