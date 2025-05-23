# postmerger

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13220424.svg)](https://zenodo.org/doi/10.5281/zenodo.13220424)

`postmerger` provides surrogate fits for binary black-hole remnants.

### Available models

- `3dq8_20M`: Ringdown amplitudes from non-precessing, quasi-circular black-hole binaries. Calibrated at $20M$ after the peak of the $(2,2)$ strain. See the example notebook [3dq8_20M](examples/3dq8_20M.ipynb) for usage details. Described in the following paper: [2408.05276](https://arxiv.org/abs/2408.05276) .

- `Prec6dq10_20M`, `Prec7dq10_20M`: Ringdown amplitudes from precessing, quasi-circular black-hole binaries. Calibrated at $20M$ after $t_{\rm EMOP}$. See the example notebook [Prec_models](examples/Prec_models.ipynb) for usage details. Described in the following paper: [2504.17021](https://arxiv.org/abs/2504.17021) .

## Installation

### From source

```
git clone https://github.com/cpacilio/postmerger.git
cd postmerger
pip install .
```

## Basic usage

```python
import postmerger
```

### List of available fits

```python
postmerger.allowed_fits
>>> ['3dq8_20M','Prec6dq10_20M', 'Prec7dq10_20M']
```

### Load fit

```python
fitname = '3dq8_20M'
fit = postmerger.load_fit(filename=fitname, download = False)
```
Set `downlaod = True` to download files containing the pre-trained precessing models from the Zenodo record (~500MB each).

### Predict amplitudes and phases

Predicting amplitudes and phases is performed through the `predict_amp` and `predict_phase` methods of `fit`. The arguments are model-specific. See the notebooks below for examples of specific models:

- [3dq8_20M](examples/3dq8_20M.ipynb)
- [Prec_models](examples/Prec_models.ipynb)

### Evaluate quasi-normal modes of Kerr black holes

We provide a QNM evaluator. See the notebook [qnm_Kerr](examples/qnm_Kerr.ipynb) for advanced usage.

```python
## mass and spin
mass = 67
spin = 0.67

## mode
mode = (2,2,0)

## evaluate in SI units
f, tau = postmerger.qnm_Kerr(mass,spin,mode,SI_units=True)

## results
print('frequency (Hz): ',f)
print('damping time (s): ',tau)
>>> frequency (Hz): 250.71404280475124
>>> damping time (s): 0.004032098030215414
```

### Evaluate spherical-spheroidal mixing coefficients

We provide a mixing-coefficient evaluator. See the notebook [spherical_spheroidal_mixing](examples/spherical_spheroidal_mixing.ipynb) for further details.

```python
## evaluate mu_{2320}

## spherical-harmonic indices
lm = (3,2)

## spheroidal-harmonic indices
mode = (2,2,0)

## final spin
spin = 0.68

mu_re, mu_im = postmerger.spherical_spheroidal_mixing(lm,mode,spin)

## results
print(mu_re+1j*mu_im)
>>> (0.0665939069543019+0.011046238081249502j)
```

### Compute final mass and final spin

We provide functions to compute final mass and final spin from binary parameters.  See the notebook [final_mass_spin](examples/final_mass_spin.ipynb) for further details.

__NOTE:__ _These fits are closed-form approximations calibrated on non-precessing simulations. The `final_spin` routine adopts an heuristic approximation to generalize to precessing spins, which is not necessarily accurate. Therefore, when performing numerical ringdown fits and calibration of GPR models, we used final masses and spins as provided by the SXS metadata._

```python
## (we set a binary with anti-aligned spins ending into a black-hole with final spin pointing downward)

## initial masses (mass1>=mass2 is required)
mass1 = 25
mass2 = 5

## initial spins (magnitudes)
spin1 = 0.9
spin2 = 0.1

## angle between spins and z-direction
## [0,pi/2] is positive-z direction
## [pi/2,pi] is negative-z direction
beta = np.pi
gamma = 0.

## relative orientation between spin1 and spin2
## here, since the spins are (anti-)aligned, alpha is forced to be arccos(cos(beta)*cos(gamma))
alpha = np.arccos(np.cos(beta)*np.cos(gamma))

## compute final mass and final spin
massf = postmerger.final_mass(mass1,mass2,spin1,spin2,alpha,beta,gamma)
spinf, thetaf = postmerger.final_spin(mass1,mass2,spin1,spin2,,alpha,beta,gamma,return_angle=True)

## results
print('final mass: ',massf)
print('final spin: ',spinf)
print('final orientation: ',np.cos(thetaf))
>>> final mass: 29.62197225289648
>>> final spin: 0.12753062487767092
>>> final orientation: -1.0
```

## Reproducibility

To help reproducibility, we provide data on which our model were trained:

- [data_for_3dq8_20M](examples/data_for_3dq8_20M.ipynb)
- [data_for_prec_models](examples/data_for_prec_models.ipynb)

## Citation

If you use `postmerger` in your work, please cite the following entries:

```latex
@article{Pacilio:2024tdl,
    author = {Pacilio, Costantino and Bhagwat, Swetha and Nobili, Francesco and Gerosa, Davide},
    title = {{postmerger: A flexible mapping of ringdown amplitudes for non-precessing binary black holes}},
    eprint = {2408.05276},
    archivePrefix = {arXiv},
    primaryClass = {gr-qc},
    month = {8},
    year = {2024}
}
```


```latex
@software{pacilio_2024_13220424,
  author       = {Pacilio, Costantino and
                  Bhagwat, Swetha  and
                  Nobili, Francesco  and
                  Gerosa, Davide},
  title        = {postmerger},
  month        = aug,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v0.0.2},
  doi          = {10.5281/zenodo.13220424},
  url          = {https://doi.org/10.5281/zenodo.13220424}
}
```

```latex
@article{Nobili:2025ydt,
    author = "Nobili, Francesco and Bhagwat, Swetha and Pacilio, Costantino and Gerosa, Davide",
    title = "{Ringdown mode amplitudes of precessing binary black holes}",
    eprint = "2504.17021",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    month = "4",
    year = "2025"
}
```



## Coming soon

- Surrogate fits for the final mass and final spin of the remnant 
- Full API