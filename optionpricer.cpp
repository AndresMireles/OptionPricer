#include "OptionPricer.h"
#include <cmath>

namespace project {

// Constructor
OptionPricer::OptionPricer(
    Option option,
    int n,
    int k,
    double S0,
    std::vector<double> riskFreeRate,
    double T0
) {

    if (T0 < 0) {
        throw std::invalid_argument("Computation time must be greater than 0.");
    } 
    if (S0 < 0) {
        throw std::invalid_argument("Spot price must be greater than 0.");
    }  
    if ((n < 0) || (k < 0)) {
        throw std::invalid_argument("Grid parameters must be greater than 0.");
    } 
    if (T0 > option.getMaturity()) {
        throw std::invalid_argument("Computation date must be earlier than maturity date");
    }

    option_ = option;
    n_ = n;
    k_ = k;
    S0_ = S0;
    riskFreeRate_ = riskFreeRate;
    T0_ = T0;
}

// Method to calculate the option price
double OptionPricer::computePrice() const {
    // Placeholder for computation logic
    return 0.0;
}

// The method for setting up the grid
void OptionPricer::setupGrid() {
    // Set up the grid with n_ + 1 rows and k_ + 1 columns, and fill it with 0
    grid_.resize(n_ + 1, std::vector<double>(k_ + 1, 0.0));

    // Define time and spot intervals based on n_, k_, S0_, and T0_
    double dt = (option_.getMaturity() - T0_) / k_;
    double dS = 2 * S0_ / n_; // Creates a symmetric spot price range around S0_

    // Populate spot prices for each step
    spotPrices_.resize(n_ + 1);
    for (int i = 0; i <= n_; ++i) {
        spotPrices_[i] = i * dS; // Spot prices for each grid row
    }

    // Populate time steps for each time interval
    timeSteps_.resize(k_ + 1);
    for (int j = 0; j <= k_; ++j) {
        timeSteps_[j] = T0_ + j * dt; // Time steps for each column in the grid
    }
}

// Method for setting up the initial conditions
void OptionPricer::initializeConditions() {
    // Retrieve option details
    double strike = option_.getStrike();
    double maturity = option_.getMaturity();
    bool isCall = (option_.getOptionType() == "Call");

    // Initial condition at maturity (t = T)
    for (int i = 0; i <= n_; ++i) {
        double spot = spotPrices_[i];
        if (isCall) {
            // Call option payoff at maturity
            grid_[i][k_] = std::max(spot - strike, 0.0);
        } else {
            // Put option payoff at maturity
            grid_[i][k_] = std::max(strike - spot, 0.0);
        }
    }

    // Boundary condition for S = 0 (lowest spot price)
    for (int j = 0; j <= k_; ++j) {
        double t = timeSteps_[j];
        if (isCall) {
            grid_[0][j] = 0.0; // Call option has no value if S = 0
        } else {
            grid_[0][j] = strike * exp(-riskFreeRate_[0] * (maturity - t)); // Put option value at S = 0
        }
    }

    // Boundary condition for very high S (highest spot price)
    for (int j = 0; j <= k_; ++j) {
        double t = timeSteps_[j];
        if (isCall) {
            grid_[n_][j] = spotPrices_[n_] - strike * exp(-riskFreeRate_[0] * (maturity - t));
        } else {
            grid_[n_][j] = 0.0; // Put option has no value if S is very high
        }
    }
}

void OptionPricer::performCalculations() {
    // Placeholder for finite difference calculations
}

} // namespace project
