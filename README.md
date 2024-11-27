# Option Pricing Library

## Overview

The **Option Pricing Library** is a C++ project designed to model and price financial options using both analytical and numerical methods. It provides a robust framework for representing financial options and calculating their prices and Greeks (sensitivities) using the Black-Scholes model and Partial Differential Equation (PDE)-based finite difference methods.

## Features

- **Option Representation**: Model financial options with customizable parameters including option type (Call or Put), exercise type (European or American), strike price, maturity, and volatility.
- **Pricing Methods**:
  - **Black-Scholes Model**: Analytical pricing for European options.
  - **Finite Difference PDE Solver**: Numerical pricing for both European and American options.
- **Greeks Calculation**: Compute sensitivities such as Delta, Gamma, Theta, Vega, and Rho using both analytical (Black-Scholes) and numerical (Finite Difference) methods.
- **Parameter Analysis**: Evaluate option prices and Greeks over a range of parameter values (e.g., varying spot price, maturity, or volatility).
- **Exercise Boundary Determination**: Calculate and export the exercise boundary for American options.
- **Comprehensive Documentation**: Detailed inline documentation using Doxygen-style comments.
