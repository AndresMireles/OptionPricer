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
        std::vector<double> riskFreeRates,
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

private:
    int n_; // Number of spot intervals
    int k_; // Number of time intervals
    double S0_;
    std::vector<double> riskFreeRates_;
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
    void initializeConditions(double maturity, std::vector<double> riskFreeRates);
    void performCalculations(std::vector<double> riskFreeRates, double volatility);

    // Auxiliary functions for the analytical solutions
    double normCDF(double x) const;
    double normPDF(double x) const;

    // Method to calculate the option price using Black-Scholes (only valid for European options)
    double computePriceBS();

    // Method to calculate the option price using finite differences
    double computePricePDE(double S0, double maturity, std::vector<double> riskFreeRates, double volatility);

    // Method to analytically compute a greek that is provided as input (only valid for European options)
    double computeGreekBS(const std::string greek);

    // Method to numerically compute a greek that is provided as input
    double computeGreekNumerical(const std::string greek);

};

} // namespace project

#endif // OPTIONPRICER_H
