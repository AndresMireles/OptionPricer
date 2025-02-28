# Option Pricing Library

## Overview

The **Option Pricing Library** is a C++ project designed to model and price financial options using both analytical and numerical methods. It provides a framework for representing financial options and calculating their prices and Greeks (sensitivities) using the Black-Scholes model and Partial Differential Equation (PDE)-based finite difference methods.

## Features

- **Option Representation**: Model financial options with customizable parameters including option type (Call or Put), exercise type (European or American), strike price, maturity, and volatility.
- **Pricing Methods**:
  - **Black-Scholes Model**: Analytical pricing for European options (using the Black '76 representation with term structure of interest rates).
  - **Finite Difference PDE Solver**: Numerical pricing for both European and American options.
- **Greeks Calculation**: Compute sensitivities such as Delta, Gamma, Theta, Vega, and Rho using both analytical (Black-Scholes) and numerical (Finite Difference) methods.
- **Parameter Analysis**: Evaluate option prices and Greeks over a range of parameter values (e.g., varying spot price, maturity, or volatility).
- **Exercise Boundary Determination**: Calculate and export the optimal exercise boundary for American options.

![image](https://github.com/user-attachments/assets/26bf2424-99b7-4210-a0d1-181cd070af0b)
