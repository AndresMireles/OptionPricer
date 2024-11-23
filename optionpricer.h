#ifndef OPTIONPRICER_H
#define OPTIONPRICER_H

#include "Option.h"
#include <vector>

namespace project {

/**
 * @class OptionPricer
 * @brief Class for pricing financial options using various methods.
 *
 * The OptionPricer class provides functionality to compute option prices and Greeks
 * using analytical and numerical methods such as Black-Scholes and PDE-based finite
 * difference methods.
 */

class OptionPricer {
public:
    /**
     * @brief Constructs an OptionPricer object with the specified parameters.
     *
     * @param option The financial option to be priced.
     * @param n Number of spot steps in the finite difference grid.
     * @param k Number of time steps in the finite difference grid.
     * @param S0 Initial spot price of the underlying asset.
     * @param riskFreeTimes Vector of times at which risk-free rates are specified.
     * @param riskFreeRates Vector of corresponding risk-free rates.
     * @param dividendYield Continuous dividend yield of the underlying asset.
     * @param T0 Initial computation time (default is 0).
     *
     * @throws std::invalid_argument if any of the input parameters are invalid.
     */
    OptionPricer(
        Option option,
        int n, // Number of spot steps
        int k, // Number of time steps
        double S0,
        std::vector<double> riskFreeTimes,
        std::vector<double> riskFreeRates,
        double dividendYield,
        double T0 = 0 // Pricing date (in years)
    );

    /**
     * @brief Computes the option price using the specified method.
     *
     * @param method The pricing method to use ("Black-Scholes" or "PDE").
     * @return The computed option price.
     *
     * @throws std::invalid_argument if an invalid method is specified.
     */
    double computePrice(const std::string method);

    /**
     * @brief Compares analytical (Black-Scholes) and numerical (PDE) option prices.
     *
     * Outputs the prices and the error between them.
     */
    void comparePrices();

    /**
     * @brief Computes the specified Greek using the given method.
     *
     * @param greek The Greek to compute ("Delta", "Gamma", "Theta", "Vega", "Rho").
     * @param method The method to use ("Black-Scholes" or "Numerical").
     * @return The computed Greek value.
     *
     * @throws std::invalid_argument if an invalid Greek or method is specified.
     */
    double computeGreek(const std::string greek, const std::string method);

    /**
     * @brief Compares analytical (Black-Scholes) and numerical (Finite-Differences) Greeks.
     *
     * @param greek The Greek to compare.
     */
    void compareGreeks(const std::string greek);

    /**
     * @brief Computes option prices over a range of parameter values.
     *
     * @param param The parameter to vary ("Spot", "Maturity", or "Volatility").
     * @param paramRange The range of values for the parameter.
     * @return A vector of computed option prices.
     *
     * @throws std::invalid_argument if an invalid parameter name is provided.
     */
    std::vector<double> computePricesVector(const std::string param, const std::vector<double> paramRange);

    /**
     * @brief Computes option Greeks over a range of parameter values.
     *
     * @param greek The Greek to compute ("Delta", "Gamma", "Theta", "Vega", "Rho").
     * @param param The parameter to vary ("Spot", "Maturity", or "Volatility").
     * @param paramRange The range of values for the parameter.
     * @return A vector of computed Greek values.
     *
     * @throws std::invalid_argument if an invalid parameter or Greek name is provided.
     */
    std::vector<double> computeGreeksVector(const std::string greek, const std::string param, const std::vector<double> paramRange);

    /**
     * @brief Interpolates the risk-free rate at a given time.
     *
     * Performs linear interpolation between specified risk-free rates based on the input time `t`.
     * An optional adjustment `deltaR` can be added to the interpolated rate.
     *
     * @param t The time at which to interpolate the risk-free rate.
     * @param deltaR An optional adjustment to the interpolated rate (default is 0.0).
     * @return The interpolated risk-free rate.
     *
     * @throws std::runtime_error if the interpolation cannot be performed.
     */
    double interpolateRiskFreeRate(double t, double deltaR = 0.0);

private:
    // Member variables with brief descriptions
    int n_; ///< Number of spot intervals
    int k_; ///< Number of time intervals
    double S0_; ///< Initial spot price
    std::vector<double> riskFreeTimes_; ///< Times for risk-free rates
    std::vector<double> riskFreeRates_; ///< Corresponding risk-free rates
    double dividendYield_; ///< Continuous dividend yield
    double T0_; ///< Initial computation time
    
    double interpolateRiskFreeRate(double t, double deltaR = 0.0); // Method to interpolate the risk free rates

    // Parameters of the option
    std::string optionType_; ///< Type of the option ("Call" or "Put")
    std::string exerciseType_; ///< Exercise type ("European" or "American")
    double strike_; ///< Strike price of the option
    double maturity_; ///< Maturity time of the option
    double volatility_; ///< Volatility of the underlying asset

    // Grid to hold option values
    std::vector<std::vector<double>> grid_; ///< Finite difference grid

    // Spot and time steps
    std::vector<double> spotPrices_; ///< Discrete spot prices
    std::vector<double> timeSteps_; ///< Discrete time steps

    // Auxiliary methods for finite difference calculations
    void setupGrid(double S0, double maturity);
    void initializeConditions(double maturity, double deltaR = 0.0);
    void performCalculations(double volatility, double deltaR = 0.0);

    // Auxiliary functions for the analytical solutions
    double normCDF(double x) const; ///< Standard normal cumulative distribution function
    double normPDF(double x) const; ///< Standard normal probability density function

    // Pricing methods
    double computePriceBS(); ///< Computes price using Black-Scholes model
    double computePricePDE(double S0, double maturity, double volatility, double deltaR = 0.0); ///< Computes price using PDE

    // Greek computations
    double computeGreekBS(const std::string greek); ///< Computes Greek analytically using Black-Scholes
    double computeGreekNumerical(const std::string greek, double S0, double maturity, double volatility); ///< Computes Greek numerically using finite differences
};

} // namespace project

#endif // OPTIONPRICER_H
