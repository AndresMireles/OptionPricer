#include "option.h"
#include "optionpricer.h"

#include <iostream>
#include <string>

int main() {

    // Option parameters
    std::string optionType = "Put";
    std::string exerciseType = "European";
    double strike = 110.0;
    double maturity = 1;
    double volatility = 0.17;

    project::Option opt(optionType, exerciseType, strike, maturity, volatility);

    std::cout << opt;

    // Pricer parameters
    int n = 100000; // Number of spot steps
    int k = 1000; // Number of time steps
    double S0 = 120;
    std::vector<double> riskFreeRate(k+1, 0.07);
    double T0 = 0.0;

    project::OptionPricer pricer(opt, n, k, S0, riskFreeRate, T0);

    // double PDE_price = pricer.computePrice("PDE");

    pricer.comparePrices();
    

    return 0;
}