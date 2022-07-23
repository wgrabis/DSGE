from enum import Enum


class RunMode(Enum):
    forecastBlanchard = 'fblanchard'


class LogMode(Enum):
    debug = 'debug',
    none = 'none'
