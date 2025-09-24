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
    parameters for that variant using the class's C{get_variants}
    method. It will then call the class's C{__init__} method with those
    parameters, along with any other provided arguments, taking care of
    managing conflicts between explicitly provided arguments and variant
    parameters.

    If the C{variant} argument is not provided or is set to 'none', the
    class will be instantiated normally.

    Additionally, if the class has a C{__post_init__} method, it will be
    called after the initialization. This method can be used for any
    additional setup that needs to occur after the object is created.
    """

    def __handle_variants(
        cls: type["VariantBase"],  # type: ignore
        *args,
        variant: str | None = None,
        **kwargs,
    ) -> "VariantBase":
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
        obj = cls.__handle_variants(*args, variant=variant, **kwargs)
        if isinstance(obj, cls):
            post_init = getattr(obj, "__post_init__", None)
            if callable(post_init):
                post_init()
        return obj


class VariantBase(ABC, metaclass=VariantMeta, register=False):
    _variant: str | None

    @staticmethod
    @abstractmethod
    def get_variants() -> tuple[str, dict[str, Kwargs]]:
        """Returns a name of the default varaint and a dictionary of
        available variants.

        The keys are the variant names, and the values are dictionaries
        of parameters which can be used as C{**kwargs} for the
        constructor of a derived class.

        @rtype: tuple[str, dict[str, Kwargs]]
        @return: A tuple containing the default variant name and a
            dictionary of available variants with their parameters.
        """
        ...


def add_variant_aliases(
    variants: dict[str, Kwargs],
    aliases: dict[str, Collection[str]] | Literal["yolo"] = "yolo",
) -> dict[str, Kwargs]:
    """Adds yolo-style aliases to the variants dictionary."""
    if aliases == "yolo":
        aliases = {
            "tiny": ["t"],
            "nano": ["n"],
            "small": ["s"],
            "medium": ["m"],
            "large": ["l"],
        }
    else:
        for alias, names in aliases.items():
            for name in names:
                variants[alias] = variants[name]

    return variants
