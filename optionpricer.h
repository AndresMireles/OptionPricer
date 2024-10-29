#ifndef OPTIONPRICER_H
#define OPTIONPRICER_H

#include "Option.h"
#include <vector>

namespace project {

class OptionPricer {
public:
    // Constructor
    OptionPricer(
        const Option& option,
        int n,
        int k,
        double S0,
        const std::vector<double>& riskFreeRate,
        double T0 = 0
    );

    // Method to calculate the option price
    double computePrice() const;

private:
    // Reference to the option
    const Option& option_;
    int n_;
    int k_;
    int S0_;
    const std::vector<double>& riskFreeRate_;
    double T0_;

    // Auxiliary methods for finite difference calculations (placeholders)
    void setupGrid();
    void initializeConditions();
    void performCalculations();
};

} // namespace project

#endif // OPTIONPRICER_H
