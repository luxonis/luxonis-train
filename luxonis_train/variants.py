"""The machinery behind the ``variant`` argument of nodes and models.

A variant is a named set of constructor arguments. A class declares its
variants in `VariantBase.get_variants`. `VariantMeta` applies the
selected set when a call creates an instance. A config thus selects a
size with one word instead of a list of parameters.

"""

from abc import ABC, abstractmethod
from collections.abc import Collection
from typing import Literal

from loguru import logger
from luxonis_ml.typing import Kwargs
from luxonis_ml.utils import AutoRegisterMeta


class VariantMeta(AutoRegisterMeta):
    """Metaclass that builds an instance from a named variant.

    A call to a class with this metaclass accepts the keyword argument
    ``variant``. The argument does not reach ``__init__``. Its value
    selects how the metaclass builds the instance:

    - ``None``, ``""``, or ``"none"``: the metaclass calls ``__init__``
      with the other arguments.
    - ``"default"``: the metaclass selects the default variant from
      ``get_variants``. When ``get_variants`` raises
      ``NotImplementedError``, the metaclass logs a warning and calls
      ``__init__`` with the other arguments.
    - Any other name: the metaclass selects the variant of that name.

    For a selected variant, the metaclass stores the variant name in
    ``_variant``. Then it calls ``__init__`` with the parameters of the
    variant and the other arguments. A keyword argument of the call
    replaces the variant parameter of the same name, and the metaclass
    logs an info message for it. Without a selected variant,
    ``_variant`` keeps its class default ``None``.

    After ``__init__``, the metaclass calls the ``__post_init__`` method
    of the instance when the class defines one. The class registration
    comes from the ``AutoRegisterMeta`` base of ``luxonis_ml``.

    Example:
        >>> from luxonis_train.variants import VariantBase
        >>> class Block(VariantBase, register=False):
        ...     def __init__(self, width: int = 1):
        ...         self.width = width
        ...
        ...     @staticmethod
        ...     def get_variants():
        ...         return "n", {"n": {"width": 8}, "s": {"width": 16}}
        >>> Block(variant="default").width
        8
        >>> Block().width
        1

    """

    def __handle_variants(
        cls: type["VariantBase"],  # type: ignore
        *args,
        variant: str | None = None,
        **kwargs,
    ) -> "VariantBase":
        """Create an instance and initialize it from the variant.

        The method deletes from the variant parameters each key that
        ``kwargs`` also holds. It thus edits the dictionary that
        ``get_variants`` returned.

        Args:
            *args (``Any``): Positional arguments for ``__init__``.
            variant (str | None): The name of the variant,
                ``"default"``, or ``"none"``. ``None`` and ``""`` act as
                ``"none"``.
            **kwargs (``Any``): Keyword arguments for ``__init__``.

        Returns:
            VariantBase: The initialized instance.

        Raises:
            NotImplementedError: When ``get_variants`` raises it and
                ``variant`` is not ``"default"``.
            ValueError: When ``get_variants`` has no variant of the
                selected name.

        """
        obj = cls.__new__(cls, *args, **kwargs)
        variant = variant or "none"

        if variant == "none":
            cls.__init__(obj, *args, **kwargs)
            return obj

        try:
            default, variants = obj.get_variants()
        except NotImplementedError as e:
            if variant != "default":
                raise NotImplementedError(
                    f"'{cls.__name__}' was called with the 'variant' "
                    f"parameter set to '{variant}', but the `get_variants` "
                    "method was not implemented."
                ) from e
            default, variants = "", {}
            implemented = False
        else:
            implemented = True

        if not implemented:
            logger.warning(
                f"'{cls.__name__}' was called with the 'variant' "
                "parameter set to 'default', but the `get_variants` "
                "method was not implemented. Using default parameters."
            )
            cls.__init__(obj, *args, **kwargs)
            return obj

        if variant == "default":
            variant = default

        obj._variant = variant  # type: ignore

        if variant not in variants:
            raise ValueError(
                f"Variant '{variant}' is not available. "
                f"Available variants: {list(variants.keys())}."
            )

        params = variants[variant]

        for key in list(params.keys()):
            if key in kwargs:
                logger.info(
                    f"Overriding variant parameter '{key}' with "
                    f"explicitly provided value `{kwargs[key]}`."
                )
                del params[key]

        cls.__init__(obj, *args, **kwargs, **params)
        return obj

    def __call__(
        cls: type["VariantBase"],  # type: ignore
        *args,
        variant: str | None = None,
        **kwargs,
    ):
        """Build an instance, then call its ``__post_init__``.

        Args:
            *args (``Any``): Positional arguments for ``__init__``.
            variant (str | None): The name of the variant,
                ``"default"``, or ``"none"``. ``None`` and ``""`` act as
                ``"none"``.
            **kwargs (``Any``): Keyword arguments for ``__init__``.

        Returns:
            VariantBase: The initialized instance.

        Raises:
            NotImplementedError: When ``get_variants`` raises it and
                ``variant`` is not ``"default"``.
            ValueError: When ``get_variants`` has no variant of the
                selected name.

        """
        obj = cls.__handle_variants(*args, variant=variant, **kwargs)
        if isinstance(obj, cls):
            post_init = getattr(obj, "__post_init__", None)
            if callable(post_init):
                post_init()
        return obj


class VariantBase(ABC, metaclass=VariantMeta, register=False):
    """Base class for classes that `VariantMeta` builds from variants.

    A subclass declares its variants in `get_variants`. A subclass
    without variants overrides `get_variants` to raise
    ``NotImplementedError``, as `BaseNode` does.

    Attributes:
        _variant (str | None): The name of the selected variant.
            `VariantMeta` sets it only when a call selects a variant.
            Otherwise it is ``None``.

    """

    _variant: str | None = None

    @staticmethod
    @abstractmethod
    def get_variants() -> tuple[str, dict[str, Kwargs]]:
        """Get the default variant name and the available variants.

        The keys of the dictionary are the variant names. Each value
        holds keyword arguments for the constructor of the class.
        `VariantMeta` passes the arguments of the selected variant to
        ``__init__``. The default variant name must be a key of the
        dictionary.

        An implementation must return new dictionaries on each call,
        because `VariantMeta` deletes the keys that a call replaces.

        Returns:
            ``tuple[str, dict[str, Kwargs]]``: The default variant name,
            and the variants with their constructor arguments.

        Raises:
            NotImplementedError: When the class has no variants.

        """
        ...


def add_variant_aliases(
    variants: dict[str, Kwargs],
    aliases: dict[str, Collection[str]] | Literal["yolo"] = "yolo",
) -> dict[str, Kwargs]:
    """Add alias names to a dictionary of variants.

    For each variant name in ``aliases`` that ``variants`` holds, the
    function adds an entry for each alias. The alias entry is the same
    dictionary object as the entry of the variant, not a copy. An alias
    replaces an entry of the same name. The function skips a name that
    ``variants`` does not hold.

    Args:
        variants (``dict[str, Kwargs]``): The variants, keyed by name.
            The function adds the aliases to this dictionary in place.
        aliases (``dict[str, Collection[str]] | Literal["yolo"]``): Each
            variant name mapped to its aliases. ``"yolo"`` maps
            ``"tiny"``, ``"nano"``, ``"small"``, ``"medium"``, and
            ``"large"`` to their first letters, and each first letter
            back to its full name.

    Returns:
        ``dict[str, Kwargs]``: The ``variants`` dictionary itself, with
        the aliases.

    Example:
        >>> add_variant_aliases({"n": {"width": 8}, "l": {"width": 64}})
        {'n': {'width': 8}, 'l': {'width': 64},
         'nano': {'width': 8}, 'large': {'width': 64}}
        >>> add_variant_aliases(
        ...     {"a": {"x": 1}}, {"a": ["alpha"], "b": ["beta"]}
        ... )
        {'a': {'x': 1}, 'alpha': {'x': 1}}

    """
    if aliases == "yolo":
        aliases = {
            "tiny": ["t"],
            "nano": ["n"],
            "small": ["s"],
            "medium": ["m"],
            "large": ["l"],
            "t": ["tiny"],
            "n": ["nano"],
            "s": ["small"],
            "m": ["medium"],
            "l": ["large"],
        }
    for name, alias_names in aliases.items():
        if name in variants:
            for alias in alias_names:
                variants[alias] = variants[name]

    return variants
