from typing import TypeVar

from passporteye.mrz.text import MRZ

ImageSize = tuple[int, int]
T = TypeVar("T", int, float)
BoxLocation = tuple[T, T, T, T]
PassportEyeMRZ = MRZ
