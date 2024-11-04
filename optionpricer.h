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
        int n,
        int k,
        double S0,
        std::vector<double> riskFreeRate,
        double T0 = 0
    );

    // Method to calculate the option price
    double computePrice() const;

private:
    // Reference to the option
    Option option_;
    int n_; // Number of Spot intervals
    int k_; // Number of time intervals
    double S0_;
    std::vector<double> riskFreeRate_;
    double T0_;

    // Grid to hold option values
    std::vector<std::vector<double>> grid_;

    // Spot and time steps
    std::vector<double> spotPrices_;
    std::vector<double> timeSteps_;

    // Auxiliary methods for finite difference calculations (placeholders)
    void setupGrid();
    void initializeConditions();
    void performCalculations();
};

} // namespace project

#endif // OPTIONPRICER_H
