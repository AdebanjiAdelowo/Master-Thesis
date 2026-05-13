# Abstract

**Optimal Mixing of Passive Scalars via the Lin--Thiffeault--Doering Velocity: A Pseudo-Spectral Python Implementation**

Adebanji Adelowo

---

This thesis investigates the optimal mixing of passive scalars in incompressible fluid flow using the instantaneous-optimal (greedy) velocity field derived by Lin, Thiffeault & Doering (2011). The central objective is to numerically characterise the decay rate of the $H^{-1}$ mix norm under the LTD optimal stirring strategy and to examine how the mixing timescale depends on the spatial scale of the initial scalar distribution.

A full Python implementation of the pseudo-spectral simulation is developed, porting an existing MATLAB codebase to an open, reproducible framework. The scalar transport equation is solved on the two-dimensional torus $[0,1]^2$ using fast Fourier transforms for spatial discretisation and an adaptive Dormand--Prince RK45 scheme for time integration. Four families of initial conditions --- parameterised by a support-size parameter $a$ --- are studied, with $L^p$ norm conservation used as a resolution diagnostic and stopping criterion.

Numerical results confirm that the $H^{-1}$ mix norm decays approximately exponentially under LTD stirring, consistent with theoretical predictions. For a representative case ($a = 0.5$), the fitted decay rate is $\lambda \approx 0.44$ (mixing timescale $\tau \approx 2.27$). Across eight values of $a \in [0.5,\, 0.9375]$, the mixing timescale grows as $\lambda \propto a^{-1.78}$, moderately steeper than the theoretically predicted $a^{-1}$ scaling; this discrepancy is attributed to finite grid resolution constraining access to the asymptotic regime.

The complete Python codebase, Jupyter notebooks, and automated figure generation pipeline are contributed as reproducible research artefacts, providing a foundation for future studies at higher spectral resolution and for comparison with non-greedy global optimisation strategies.

**Keywords:** passive scalar mixing, $H^{-1}$ mix norm, pseudo-spectral methods, optimal stirring, Lin--Thiffeault--Doering velocity, incompressible fluid mechanics
