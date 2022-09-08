from enum import Enum


class RunMode(Enum):
    forecastBlanchard = 'fblanchard'
    estimation = 'estimate'
    testEstimation = 'test_estimate'
    forecastEstimation = 'festimate'


class LogMode(Enum):
    debug = 'debug',
    none = 'none'
