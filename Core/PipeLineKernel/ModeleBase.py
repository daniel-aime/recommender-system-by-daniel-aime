from .VectorBiais import *
import pandas as pd
import numpy as np


class ModeleBase:

    def __init__(self):
        self.predict_init = None

    def fit(self):
        pass
