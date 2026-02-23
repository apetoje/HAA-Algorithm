# HAA-Algorithm
Hierarchical Approximation Algorithm

Author
Aleksandar Petojević
Faculty of Education, University of Novi Sad,
Podgoričcka 4, 25000 Sombor, Republic of Serbia
ORCID: 0000-0003-1491-165X

## HAA Algorithm for the Fine-Structure Constant

## Description
Python implementation of the Hierarchical Approximation Algorithm (HAA) for approximating the reciprocal fine-structure constant α⁻¹ = 137.035999177 (CODATA 2022).

## Parameters

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `alpha_inverse` | `137.035999177` | Target value – the reciprocal fine-structure constant (CODATA 2022) |
| `S0_initial` | `137` | Initial approximation – the integer part (floor) of the target value |
| `eps_target` | `9.99e-17` | Target relative precision – the algorithm stops when the relative error falls below this value |
| `max_restarts` | `30` | Maximum number of restarts – high enough to guarantee convergence |
| `max_q` | `80` | Maximum hierarchical level – the largest index q in the basis {κ_q} used |

## Running the Code
```bash
python haa_alpha.py
