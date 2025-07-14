# ExoMUSE
**Exoplanet Modelling and Understanding for Science and Education**

## Capabilities
* Radial Velocity Modelling
* Transit Modelling
* Radial Velocity - Transit Joint Modelling
* Rositter-Mclaughlin (RM) effect Modelling
* Radial Velocity - Astrometry Joint Modelling (Work in Progress)

## Package Structure
* **ExoMUSE**
  * Contains the Python codes of the ExoMUSE package.
* **data**
  * Contains the example datasets alongside the datasets of TOI-2431 b.
* **notebooks**
  * Contains notebook tutorials on how to use ExoMUSE for modelling alongside the analyses conducted for the [discovery paper of TOI-2431 b](https://arxiv.org/abs/2507.08464).
* **nsstools**
  * Contains the `nsstools` code.

## Dependencies
* nsstools (`pip install nsstools`)
* PyDE (`pip install pyde`)
* emcee (`pip install emcee`)
* batman (`pip install batman-package`)
* radvel (`pip install radvel`)
* corner (`pip install corner`)
* astropy (`pip install astropy`)
