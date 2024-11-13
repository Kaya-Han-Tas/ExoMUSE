# nsstools
This python tools has two methods that applies on nss_two_body_orbit [Gaia DR3](https://www.cosmos.esa.int/web/gaia/data-release-3) solutions
- covmat: for all kind of nss_solution_type, converts the correlation matrix+uncertainties to the covariance matrix of the solution
- campbell: for a NSS solution that is either astrometric Orbital* or AstroSpectroSB1, converts the Thiele-Innes orbital elements to the Campbell elements and propagates the uncertainties.
            Ref: Halbwachs et al., 2022, Gaia Data Release 3. Astrometric binary star processing, Astronomy and Astrophysics, Appendix A
input: dataframe 
output: dataframe

A R version is available [here]( https://gricad-gitlab.univ-grenoble-alpes.fr/ipag-public/gaia/nsstools).

## Installation

### with pip
pip3 install --user nsstools

### with setup
python3 setup.py install

## Usage

See the [notebook](https://gitlab.obspm.fr/gaia/nsstools/-/blob/main/nss.ipynb)

```python3
import pandas as pd
from nsstools import NssSource

nss = pd.read_csv("tests/data/nss_two_body_orbit_sample.csv.gz")
source_index = 0 # position of the source in the csv file

source = NssSource(nss, indice=source_index)
print(source.covmat())
print(source.campbell())

```

## Authors and acknowledgment
Authors:  Nicolas Leclerc from a code by Jean-Louis Halbwachs and Carine Babusiaux.
Reference: Halbwachs et al., 2022, Gaia Data Release 3. Astrometric binary star processing, Astronomy and Astrophysics, Appendix A and B.
R version: https://gricad-gitlab.univ-grenoble-alpes.fr/ipag-public/gaia/nsstools
