# ------------------------------------------------------------------------------
#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see http://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#
# ------------------------------------------------------------------------------

import dataclasses
from dataclasses import make_dataclass
import re
import random
from argparse import ArgumentParser, ArgumentTypeError
from collections import namedtuple
from enum import Enum
from typing import Any, Iterable, List, NewType, Optional, Tuple, Union

DataClass = NewType("DataClass", Any)
DataClassType = NewType("DataClassType", Any)

def create_id(num_letters=4, num_numbers=3):
    random_id = ""
    for _ in range(num_letters):
        random_id += chr(random.randint(97, 122))
    for _ in range(num_numbers):
        random_id += str(random.randint(0, 9))

    return random_id


def merge_args(dataclasses, class_name="Args"):
    """Merge multiple dataclasses into one"""
    fields, args = {}, {}
    for dc in dataclasses:
        fields.update(dc.__dataclass_fields__)
        args.update(dc.__dict__)

    return make_dataclass(cls_name="Args", fields=fields)(**args)


def dict_to_dataclass(input_dict, class_name="Args"):
    return make_dataclass(cls_name=class_name, fields=input_dict.keys())(**input_dict)


# From https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def string_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )


class DataClassArgumentParser(ArgumentParser):
    """
    This subclass of `argparse.ArgumentParser` uses type hints on dataclasses to generate arguments.

    The class is designed to play well with the native argparse. In particular, you can add more (non-dataclass backed)
    arguments to the parser after initialization and you'll get the output back after parsing as an additional
    namespace.

    Adapted from HuggingFace HFArgumentParser
    """

    dataclass_types: Iterable[DataClassType]

    def __init__(self, dataclass_types: Union[DataClassType, Iterable[DataClassType]], **kwargs):
        """
        Args:
            dataclass_types:
                Dataclass type, or list of dataclass types for which we will "fill" instances with the parsed args.
            kwargs:
                (Optional) Passed to `argparse.ArgumentParser()` in the regular way.
        """
        super().__init__(**kwargs)
        if dataclasses.is_dataclass(dataclass_types):
            dataclass_types = [dataclass_types]
        self.dataclass_types = dataclass_types
        for dtype in self.dataclass_types:
            self._add_dataclass_arguments(dtype)

    def _add_dataclass_arguments(self, dtype: DataClassType):
        for field in dataclasses.fields(dtype):
            if not field.init:
                continue
            field_name = f"--{field.name}"
            kwargs = field.metadata.copy()
            # field.metadata is not used at all by Data Classes,
            # it is provided as a third-party extension mechanism.
            if isinstance(field.type, str):
                raise ImportError(
                    "This implementation is not compatible with Postponed Evaluation of Annotations (PEP 563),"
                    "which can be opted in from Python 3.7 with `from __future__ import annotations`."
                    "We will add compatibility when Python 3.9 is released."
                )
            typestring = str(field.type)
            for prim_type in (int, float, str):
                for collection in (List,):
                    if (
                        typestring == f"typing.Union[{collection[prim_type]}, NoneType]"
                        or typestring == f"typing.Optional[{collection[prim_type]}]"
                    ):
                        field.type = collection[prim_type]
                if (
                    typestring == f"typing.Union[{prim_type.__name__}, NoneType]"
                    or typestring == f"typing.Optional[{prim_type.__name__}]"
                ):
                    field.type = prim_type

            if isinstance(field.type, type) and issubclass(field.type, Enum):
                kwargs["choices"] = [x.value for x in field.type]
                kwargs["type"] = type(kwargs["choices"][0])
                if field.default is not dataclasses.MISSING:
                    kwargs["default"] = field.default
                else:
                    kwargs["required"] = True
            elif field.type is bool or field.type == Optional[bool]:
                if field.default is True:
                    self.add_argument(f"--no_{field.name}", action="store_false", dest=field.name, **kwargs)

                # Hack because type=bool in argparse does not behave as we want.
                kwargs["type"] = string_to_bool
                if field.type is bool or (field.default is not None and field.default is not dataclasses.MISSING):
                    # Default value is True if we have no default when of type bool.
                    default = True if field.default is dataclasses.MISSING else field.default
                    # This is the value that will get picked if we don't include --field_name in any way
                    kwargs["default"] = default
                    # This tells argparse we accept 0 or 1 value after --field_name
                    kwargs["nargs"] = "?"
                    # This is the value that will get picked if we do --field_name (without value)
                    kwargs["const"] = True
            elif (
                hasattr(field.type, "__origin__") and re.search(r"^typing\.List\[(.*)\]$", str(field.type)) is not None
            ):
                kwargs["nargs"] = "+"
                kwargs["type"] = field.type.__args__[0]
                assert all(
                    x == kwargs["type"] for x in field.type.__args__
                ), "{} cannot be a List of mixed types".format(field.name)
                if field.default_factory is not dataclasses.MISSING:
                    kwargs["default"] = field.default_factory()
                elif field.default is dataclasses.MISSING:
                    kwargs["required"] = True
            else:
                kwargs["type"] = field.type
                if field.default is not dataclasses.MISSING:
                    kwargs["default"] = field.default
                elif field.default_factory is not dataclasses.MISSING:
                    kwargs["default"] = field.default_factory()
                else:
                    kwargs["required"] = True
            self.add_argument(field_name, **kwargs)

    def parse_dict(self, args: dict) -> Tuple[DataClass, ...]:
        """
        Parses a dict and populating the dataclass types.
        """
        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype) if f.init}
            inputs = {k: v for k, v in args.items() if k in keys}
            obj = dtype(**inputs)
            outputs.append(obj)
        return (*outputs,)
