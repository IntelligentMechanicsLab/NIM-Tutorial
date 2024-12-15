# Neural-Integrated Meshfree (NIM) Method: A differentiable programming-based hybrid solver | CMAME, EWC

### [Paper1](https://www.sciencedirect.com/science/article/pii/S0045782524002809), [Paper2](https://arxiv.org/abs/2407.11183)

Honghui Du, QiZhi He, Binyao Guo<br>
University of Minnesota<br>

![NIM](docs/architecture.png)
![example](docs/result_example.png)
## Overview
A differentiable programming neural integrated meshfree (A.k.a. Differentiable Meshfree) solver based on the JAX framework, designed for both forward and inverse modeling of elastic/inelastic materials. This repository supports the accompanying paper with both data and code.
## Requirements

Python libraries required:
- jax
- tqdm
- scipy
- jaxopt
- tqdm

## Installation
To install the required Python libraries, run the following command:

pip install jax tqdm scipy jaxopt

## Tutorial Examples

### 1D Hyperelasticity using V-NIM
Explore the 1D Hyperelasticity model using the V-NIM method provided below:
- **[Code for NIM/c](1D_hyperelasticity/NIM-C_1D_hyperelasticity_Tutorial.ipynb)**

- **[Code for NIM/h](1D_hyperelasticity/NIM-H_1D_hyperelasticity_Tutorial.ipynb)**

### Forward and inverse modeling for time dependent proglem (advection diffusion equation) using S-NIM

- **[Forward modeling](Advection_diffusion_equation_forward/JAX_SNIM_ADE_forward.py)**
![Pig 1](Results\ADE_forward\81_81\loos_plot.png) ![Pig 2](Results\ADE_forward\81_81\exact_t_all_t.png) ![Pig 2](Results\ADE_forward\81_81\prediction_t_all_t.png)

- **[Inverse modeling](Advection_diffusion_equation_inverse/JAX_SNIM_ADE_inverse.py)**

More examples demonstrating the application of the NIM method including (operator learning, elastoplasticity modeling) will be released soon. Stay tuned for updates.

## Citation

```bibtex
@article{du2024neural,
  title={Neural-Integrated Meshfree (NIM) Method: A differentiable programming-based hybrid solver for computational mechanics},
  author={Du, Honghui and He, QiZhi},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={427},
  pages={117024},
  year={2024},
  publisher={Elsevier}
}

@article{du2024differentiable,
  title={Differentiable Neural-Integrated Meshfree Method for Forward and Inverse Modeling of Finite Strain Hyperelasticity},
  author={Du, Honghui and Guo, Binyao and He, QiZhi},
  journal={arXiv preprint arXiv:2407.11183},
  year={2024}
}

```
