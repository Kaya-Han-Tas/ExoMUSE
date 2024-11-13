import unittest
import math
from numpy import NaN
import pandas as pd
from nsstools import NssSource

class NssToolsTest(unittest.TestCase):

    def setUp(self) -> None:
        loaded_data = pd.read_csv("tests/data/nss_two_body_orbit_sample.csv.gz")
        self._source = NssSource(loaded_data, indice=0)

    def test_covmat(self):
        covmat = self._source.covmat()
        self.assertTrue(abs(covmat.loc["ra", "ra"] - 0.004181) <= 1e-6)
        self.assertTrue(abs(covmat.loc["t_periastron", "period"] - (-1573.314346)) <= 1e-6)

    def test_campbell(self):
        campbell = self._source.campbell()
        self.assertTrue(abs(campbell["a0"][0] - 0.861758) <= 1e-6)
        self.assertTrue(abs(campbell["inclination"][0] - 75.573712) <= 1e-6)
        self.assertTrue(abs(campbell["arg_periastron"][0] - 63.079074) <= 1e-6)
        self.assertTrue(abs(campbell["nodeangle"][0] - 47.486649) <= 1e-6)
        self.assertTrue(math.isnan(campbell["a1"][0]))
        self.assertTrue(abs(campbell["a0_error"][0] - 0.148773) <= 1e-6)
        self.assertTrue(abs(campbell["inclination_error"][0] - 3.439028) <= 1e-6)
        self.assertTrue(abs(campbell["arg_periastron_error"][0] - 10.823528) <= 1e-6)
        self.assertTrue(abs(campbell["nodeangle_error"][0] - 2.28581) <= 1e-6)
        self.assertTrue(math.isnan(campbell["a1_error"][0]))
        self.assertEquals(campbell["source_id"][0], 5706079252076583427)


if __name__ == '__main__':
    unittest.main()