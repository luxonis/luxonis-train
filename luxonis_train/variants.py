from abc import ABC, abstractmethod
from collections.abc import Collection
from typing import Literal

from loguru import logger
from luxonis_ml.typing import Kwargs
from luxonis_ml.utils.registry import AutoRegisterMeta


class VariantMeta(AutoRegisterMeta):
    """A metaclass for classes that support variants.

    When a class with this metaclass is instantiated with the 'variant'
    keyword argument, the metaclass will look up the corresponding
    parameters for that variant using the class's ``get_variants``
    method. It will then call the class's ``__init__`` method with those
    parameters, along with any other provided arguments, taking care of
    managing conflicts between explicitly provided arguments and variant
    parameters.

    If the ``variant`` argument is not provided or is set to ``"none"``,
    the class will be instantiated normally.

    Additionally, if the class has a ``__post_init__`` method, it will
    be called after the initialization. This method can be used for any
    additional setup that needs to occur after the object is created.

    """

    def __handle_variants(
        cls: type["VariantBase"],  # type: ignore
        *args,
        variant: str | None = None,
        **kwargs,
    ) -> "VariantBase":
        """Create an instance with variant parameters merged into
        kwargs.

        Args:
            *args (Any): Positional arguments forwarded to the class
                constructor.
            variant (str | None): Variant name, ``"default"``, or ``None``.
            **kwargs (Any): Keyword arguments forwarded to the class
                constructor.

        Returns:
            VariantBase: Created instance.

        Raises:
            NotImplementedError: If a non-default variant is requested but
                the class does not implement ``get_variants``.
            ValueError: If the requested variant is not available.

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
                    "method was not implented."
                ) from e
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
        """Create an instance and run ``__post_init__`` when available.

        Args:
            *args (Any): Positional arguments forwarded to the class
                constructor.
            variant (str | None): Variant name, ``"default"``, or ``None``.
            **kwargs (Any): Keyword arguments forwarded to the class
                constructor.

        Returns:
            VariantBase: Created instance.

        """
        obj = cls.__handle_variants(*args, variant=variant, **kwargs)
        if isinstance(obj, cls):
            post_init = getattr(obj, "__post_init__", None)
            if callable(post_init):
                post_init()
        return obj


class VariantBase(ABC, metaclass=VariantMeta, register=False):
    """Base class for objects constructed from named variants.

    Attributes:
        _variant (str | None): Name of the selected variant, or ``None`` when
            no variant was selected.

    """

    _variant: str | None

    @staticmethod
    @abstractmethod
    def get_variants() -> tuple[str, dict[str, Kwargs]]:
        """Get the default variant name and available variants.

        The keys are the variant names, and the values are dictionaries
        of parameters which can be used as ``**kwargs`` for the
        constructor of a derived class.

        Returns:
            tuple[str, dict[str, Kwargs]]: A tuple containing the default
            variant name and a dictionary of available variants with their
            parameters.

        """
        ...


def add_variant_aliases(
    variants: dict[str, Kwargs],
    aliases: dict[str, Collection[str]] | Literal["yolo"] = "yolo",
) -> dict[str, Kwargs]:
    """Add variant aliases to the variants dictionary.

    Args:
        variants (dict[str, Kwargs]): Variant configuration dictionary to
            mutate with alias entries.
        aliases (dict[str, Collection[str]] | Literal["yolo"]): Alias mapping
            or ``"yolo"`` for the built-in YOLO alias mapping.

    Returns:
        dict[str, Kwargs]: The input ``variants`` dictionary after alias
        handling.

    """
    skip_missing = aliases == "yolo"
    if skip_missing:
        aliases = {
            "tiny": ["t"],
            "nano": ["n"],
            "small": ["s"],
            "medium": ["m"],
            "large": ["l"],
        }
    for alias, names in aliases.items():
        for name in names:
            if skip_missing and name not in variants:
                continue
            variants[alias] = variants[name]

    return variants
