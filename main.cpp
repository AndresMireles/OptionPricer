#include "option.h"
#include "optionpricer.h"

#include <iostream>
#include <string>

int main() {

    // Option parameters
    std::string optionType = "Call";
    std::string exerciseType = "European";
    std::string exerciseType2 = "American";
    double strike = 110.0;
    double maturity = 1;
    double volatility = 0.17;

    project::Option opt(optionType, exerciseType, strike, maturity, volatility);
    project::Option optAm(optionType, exerciseType2, strike, maturity, volatility);

    std::cout << opt << std::endl;
    std::cout << optAm << std::endl;

    // Pricer parameters
    int n = 100000; // Number of spot steps
    int k = 1000; // Number of time steps
    double S0 = 120;
    std::vector<double> riskFreeTimes = {0.0, 0.5, 1.0};
    std::vector<double> riskFreeRates = {0.1, 0.05, 0.07};
    double q = 0.01;
    double T0 = 0.0;

    project::OptionPricer pricer(opt, n, k, S0, riskFreeTimes, riskFreeRates, q, T0);
    project::OptionPricer pricerAm(optAm, n, k, S0, riskFreeTimes, riskFreeRates, q, T0);

    // double PDE_price = pricer.computePrice("PDE");

    std::cout << "European price: " << pricer.computePrice("PDE") << ". American Price: " << pricerAm.computePrice("PDE") << "." << std::endl;

    pricer.comparePrices();

    pricer.compareGreeks("Delta");
    pricer.compareGreeks("Gamma");
    pricer.compareGreeks("Theta");
    pricer.compareGreeks("Vega");
    pricer.compareGreeks("Rho");    
    

    return 0;
}