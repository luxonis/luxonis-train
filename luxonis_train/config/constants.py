from typing import Final

from semver.version import Version

CONFIG_VERSION: Final[Version] = Version.parse(
    "1.0", optional_minor_and_patch=True
)
