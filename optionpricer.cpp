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
        throw std::invalid_argument("Computation date must be earlier than maturity date.");
    }
    if (riskFreeRate.size() != static_cast<std::size_t>(k)) { // We need to cast k as a size_t to compare with vect.size()
        throw std::invalid_argument("Risk free rate must be of size k.");
    }

    // Pricer parameters
    n_ = n; // Number of spot steps
    k_ = k; // Number of time steps
    S0_ = S0;
    riskFreeRate_ = riskFreeRate;
    T0_ = T0;

    // Option parameters
    optionType_ = option.getOptionType();
    exerciseType_ = option.getExerciseType();
    strike_ = option.getStrike();
    maturity_ = option.getMaturity();
    volatility_ = option.getVolatility();

}

// Method to calculate the option price
double OptionPricer::computePrice() {

    setupGrid();
    initializeConditions();
    performCalculations();

    int spot_index;

    if (n_ % 2 == 0) {
        // If n is even, S0_ lies between the two middle indices
        int lower_mid = n_ / 2 - 1;
        int upper_mid = n_ / 2;

        // Calculate the average price between the two middle grid points
        return (grid_[lower_mid][0] + grid_[upper_mid][0]) / 2.0;
    } else {
        // If n is odd, S0_ is exactly at the middle index
        spot_index = n_ / 2;
        return grid_[spot_index][0];
    }

    

}

// The method for setting up the grid
void OptionPricer::setupGrid() {
    // Set up the grid with n_ + 1 rows and k_ + 1 columns, and fill it with 0
    grid_.resize(n_ + 1, std::vector<double>(k_ + 1, 0.0));

    // Define time and spot intervals based on n_, k_, S0_, and T0_
    double dt = (maturity_ - T0_) / k_;
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

    bool isCall = (optionType_ == "Call");

    // Initial condition at maturity (t = T)
    for (int i = 0; i <= n_; ++i) {
        double spot = spotPrices_[i];
        if (isCall) {
            // Call option payoff at maturity
            grid_[i][k_] = std::max(spot - strike_, 0.0);
        } else {
            // Put option payoff at maturity
            grid_[i][k_] = std::max(strike_ - spot, 0.0);
        }
    }

    // Boundary condition for S = 0 (lowest spot price)
    for (int j = 0; j <= k_; ++j) {
        double t = timeSteps_[j];
        if (isCall) {
            grid_[0][j] = 0.0; // Call option has no value if S = 0
        } else {
            grid_[0][j] = strike_ * exp(-riskFreeRate_[0] * (maturity_ - t)); // Put option value at S = 0
        }
    }

    // Boundary condition for very high S (highest spot price)
    for (int j = 0; j <= k_; ++j) {
        double t = timeSteps_[j];
        if (isCall) {
            grid_[n_][j] = spotPrices_[n_] - strike_ * exp(-riskFreeRate_[0] * (maturity_ - t));
        } else {
            grid_[n_][j] = 0.0; // Put option has no value if S is very high
        }
    }
}

// This function is general to calls and puts
void OptionPricer::performCalculations() {

    // Time and spot step sizes
    double dt = timeSteps_[1] - timeSteps_[0];
    double dS = spotPrices_[1] - spotPrices_[0];

    // Loop backward in time from maturity to current time T0
    for (int j = k_ - 1; j >= 0; --j) {
        // Set up the tridiagonal system
        int N = n_ - 1; // Number of interior spot points
        std::vector<double> a(N, 0.0); // Lower diagonal (a_1 to a_N)
        std::vector<double> b(N, 0.0); // Main diagonal (b_1 to b_N)
        std::vector<double> c(N, 0.0); // Upper diagonal (c_1 to c_N)
        std::vector<double> d(N, 0.0); // Right-hand side

        // Loop over interior spot indices
        for (int i = 1; i < n_; ++i) {
            double S = spotPrices_[i];
            double sigma = volatility_;
            double r = riskFreeRate_[j]; // Use risk-free rate at current time step

            // Coefficients for the implicit method
            double alpha = 0.5 * sigma * sigma * S * S / (dS * dS);
            double beta = r * S / (2.0 * dS);
            double gamma = r;

            a[i - 1] = -dt * (alpha - beta);
            b[i - 1] = 1 + dt * (2 * alpha + gamma);
            c[i - 1] = -dt * (alpha + beta);

            // Right-hand side is the option value at the next time step
            d[i - 1] = grid_[i][j + 1];
        }

        // Adjust the right-hand side for boundary conditions
        // Left boundary adjustment
        d[0] -= a[0] * grid_[0][j];
        // Right boundary adjustment
        d[N - 1] -= c[N - 1] * grid_[n_][j];

        // Thomas algorithm for solving tridiagonal systems
        // Forward elimination
        for (int i = 1; i < N; ++i) {
            double m = a[i] / b[i - 1];
            b[i] = b[i] - m * c[i - 1];
            d[i] = d[i] - m * d[i - 1];
        }

        // Back substitution
        std::vector<double> x(N, 0.0); // Solution vector
        x[N - 1] = d[N - 1] / b[N - 1];
        for (int i = N - 2; i >= 0; --i) {
            x[i] = (d[i] - c[i] * x[i + 1]) / b[i];
        }

        // Update grid values for current time step
        for (int i = 1; i < n_; ++i) {
            grid_[i][j] = x[i - 1];
        }
        // The boundary values grid_[0][j] and grid_[n_][j] are already set
    }
}

} // namespace project
