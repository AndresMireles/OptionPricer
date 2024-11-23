#ifndef OPTIONPRICER_H
#define OPTIONPRICER_H

#include "Option.h"
#include <vector>

namespace project {

class OptionPricer {
public:
    // Constructor with default argument only in declaration
    OptionPricer(
        Option option,
        int n, // Number of spot steps
        int k, // Number of time steps
        double S0,
        std::vector<double> riskFreeTimes,
        std::vector<double> riskFreeRates,
        double dividendYield,
        double T0 = 0
    );

    // Method to compute the price using the method provided as input
    double computePrice(const std::string method);

    // Method to compare analytical and numerical prices
    void comparePrices();

    // Method to compute the price using the method provided as input
    double computeGreek(const std::string greek, const std::string method);

    // Method to compare analytical and numerical greeks
    void compareGreeks(const std::string greek);

    // Methods to compute PDE prices and greeks for a range of values
    std::vector<double> computePricesVector(const std::string param, const std::vector<double> paramRange);
    std::vector<double> computeGreeksVector(const std::string greek, const std::string param, const std::vector<double> paramRange);

private:
    int n_; // Number of spot intervals
    int k_; // Number of time intervals
    double S0_;
    std::vector<double> riskFreeTimes_;
    std::vector<double> riskFreeRates_;
    double dividendYield_; // Continuous dividend yield
    double interpolateRiskFreeRate(double t, double deltaR = 0.0); // Method to interpolate the risk free rates
    double T0_;

    // Parameters of the option
    std::string optionType_;
    std::string exerciseType_;
    double strike_;
    double maturity_;
    double volatility_;

    // Grid to hold option values
    std::vector<std::vector<double>> grid_;

    // Spot and time steps
    std::vector<double> spotPrices_;
    std::vector<double> timeSteps_;

    // Auxiliary methods for finite difference calculations
    void setupGrid(double S0, double maturity);
    void initializeConditions(double maturity, double deltaR = 0.0);
    void performCalculations(double volatility, double deltaR = 0.0);

    // Auxiliary functions for the analytical solutions
    double normCDF(double x) const;
    double normPDF(double x) const;

    // Method to calculate the option price using Black-Scholes (only valid for European options)
    double computePriceBS();

    // Method to calculate the option price using finite differences
    double computePricePDE(double S0, double maturity, double volatility, double deltaR = 0.0);;

    // Method to analytically compute a greek that is provided as input (only valid for European options)
    double computeGreekBS(const std::string greek);

    // Method to numerically compute a greek that is provided as input
    double computeGreekNumerical(const std::string greek, double S0, double maturity, double volatility);

};

} // namespace project

#endif // OPTIONPRICER_H
