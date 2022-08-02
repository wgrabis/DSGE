from enum import Enum


class RunMode(Enum):
    forecastBlanchard = 'fblanchard'
    estimation = 'estimate'


class LogMode(Enum):
    debug = 'debug',
    none = 'none'
