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
    if (riskFreeRate.size() != static_cast<std::size_t>(k+1)) { // We need to cast k as a size_t to compare with vect.size()
        throw std::invalid_argument("Risk free rate must be of size k + 1.");
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

// The method for setting up the grid
void OptionPricer::setupGrid(double S0, double maturity) {
    // Set up the grid with n_ + 1 rows and k_ + 1 columns, and fill it with 0
    grid_.resize(n_ + 1, std::vector<double>(k_ + 1, 0.0));

    // Define time and spot intervals based on n_, k_, S0_, and T0_
    double dt = (maturity - T0_) / k_;
    double dS = 2 * S0 / n_; // Creates a symmetric spot price range around S0_

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
void OptionPricer::initializeConditions(double maturity, std::vector<double> riskFreeRate) {

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
            grid_[0][j] = strike_ * exp(-riskFreeRate[0] * (maturity - t)); // Put option value at S = 0
        }
    }

    // Boundary condition for very high S (highest spot price)
    for (int j = 0; j <= k_; ++j) {
        double t = timeSteps_[j];
        if (isCall) {
            grid_[n_][j] = spotPrices_[n_] - strike_ * exp(-riskFreeRate[0] * (maturity - t));
        } else {
            grid_[n_][j] = 0.0; // Put option has no value if S is very high
        }
    }
}

// This function is general to calls and puts
void OptionPricer::performCalculations(std::vector<double> riskFreeRate, double volatility) {
    // Time and spot step sizes (already computed in setupGrid)
    double dt = timeSteps_[1] - timeSteps_[0];
    double dS = spotPrices_[1] - spotPrices_[0];

    bool isAmerican = (exerciseType_ == "American");
    bool isCall = (optionType_ == "Call");

    // Loop backward in time from maturity to current time T0
    for (int j = k_ - 1; j >= 0; --j) {
        // Set up the tridiagonal system
        int N = n_ - 1; // Number of interior spot points
        std::vector<double> a(N, 0.0); // Lower diagonal (a_1 to a_N)
        std::vector<double> b(N, 0.0); // Main diagonal (b_1 to b_N)
        std::vector<double> c(N, 0.0); // Upper diagonal (c_1 to c_N)
        std::vector<double> d(N, 0.0); // Right-hand side

        double r = riskFreeRate[j]; // Use risk-free rate at current time step

        // Loop over interior spot indices
        for (int i = 1; i < n_; ++i) {
            double S = spotPrices_[i];

            // Coefficients for the implicit method
            double alpha = 0.5 * volatility * volatility * S * S / (dS * dS);
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
            b[i] -= m * c[i - 1];
            d[i] -= m * d[i - 1];
        }

        // Back substitution
        std::vector<double> x(N, 0.0); // Solution vector
        x[N - 1] = d[N - 1] / b[N - 1];
        for (int i = N - 2; i >= 0; --i) {
            x[i] = (d[i] - c[i] * x[i + 1]) / b[i];
        }

        // Update grid values for current time step
        for (int i = 1; i < n_; ++i) {
            double S = spotPrices_[i];
            double intrinsicValue = isCall ? std::max(S - strike_, 0.0) : std::max(strike_ - S, 0.0);

            if (isAmerican) {
                // Apply early exercise condition for American options (whether it is worth it to execute now)
                grid_[i][j] = std::max(x[i - 1], intrinsicValue);
            } else {
                // For European options, use the computed value
                grid_[i][j] = x[i - 1];
            }
        }
        // The boundary values grid_[0][j] and grid_[n_][j] are already set
    }
}

// Method to calculate the option price
double OptionPricer::computePricePDE(double S0, double maturity, std::vector<double> riskFreeRate, double volatility) {

    setupGrid(S0, maturity);
    initializeConditions(maturity, riskFreeRate);
    performCalculations(riskFreeRate, volatility);

    int spot_index;
    // We need to choose the index of the price to return depending on the shape of our grid
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

// Function to compute the cumulative distribution function for the standard normal distribution
double OptionPricer::normCDF(double x) const {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0))); // erf function comes from cmath
}

double OptionPricer::computePriceBS() {

    if (exerciseType_ != "European") {
        throw std::logic_error("Black-Scholes price is only applicable to European options.");
    }

    double timeToMaturity = maturity_ - T0_;

    // Calculate d1 and d2
    double d1 = (std::log(S0_ / strike_) + (riskFreeRate_[k_] + 0.5 * volatility_ * volatility_) * timeToMaturity) / (volatility_ * std::sqrt(timeToMaturity));
    double d2 = d1 - volatility_ * std::sqrt(timeToMaturity);

    // Calculate N(d1) and N(d2)
    double Nd1 = normCDF(d1);
    double Nd2 = normCDF(d2);
    double N_minus_d1 = normCDF(-d1);
    double N_minus_d2 = normCDF(-d2);

    // Compute option price based on type
    double price;
    if (optionType_ == "Call") {
        price = S0_ * Nd1 - strike_ * std::exp(-riskFreeRate_[0] * (timeToMaturity)) * Nd2;
    } else { // Put
        price = strike_ * std::exp(-riskFreeRate_[0] * (timeToMaturity)) * N_minus_d2 - S0_ * N_minus_d1;
    }

    return price;
}

// Actual function that the user can use
double OptionPricer::computePrice(const std::string method) {
    if (method == "Black-Scholes") {
        return computePriceBS();
    }
    else if (method == "PDE") {
        return computePricePDE(S0_, maturity_, riskFreeRate_, volatility_);
    }
    else {
        throw std::invalid_argument("Method for computing option price must be either 'Black-Scholes' or 'PDE'.");
    }
}

// Function to print the comparison of prices
void OptionPricer::comparePrices() {
    double PDE_price = computePricePDE(S0_, maturity_, riskFreeRate_, volatility_);
    double BS_price = computePriceBS();

    double PDE_error = PDE_price - BS_price;

    std::cout << "Black-Scholes price: " << BS_price << ". PDE Price: " << PDE_price << ". PDE Error: " << PDE_error << ". PDE Relative Error: " << std::abs(PDE_error / BS_price) * 100 << "%." << std::endl;
}

// Implementation of the standard normal PDF
double OptionPricer::normPDF(double x) const {
    return (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * x * x);
}

// Method to analytically cally compute a greek that is provided as input (only valid for European options)
double OptionPricer::computeGreekBS(const std::string greek) {
    // Only applicable for European options
    if (exerciseType_ != "European") {
        throw std::logic_error("Greeks can only be computed analytically for European options.");
    }

    // Calculate time to maturity
    double timeToMaturity = maturity_ - T0_;

    // Calculate d1 and d2
    double sqrtTime = std::sqrt(timeToMaturity);
    double d1 = (std::log(S0_ / strike_) + (riskFreeRate_[0] + 0.5 * volatility_ * volatility_) * timeToMaturity) / (volatility_ * sqrtTime);
    double d2 = d1 - volatility_ * sqrtTime;

    // Calculate N(d1), N(d2), N(-d2), and N'(d1) for the analytical formulas
    double Nd1 = normCDF(d1);
    double Nd2 = normCDF(d2);
    // double N_minus_d1 = normCDF(-d1); // Not needed
    double N_minus_d2 = normCDF(-d2);
    double npd1 = normPDF(d1);

    // Compute numerical deltas
    if (greek == "Delta") {
        if (optionType_ == "Call") {return Nd1;} // Call delta
        else {return Nd1 - 1.0;} // Put delta (by Put-Call parity)
    } 
    else if (greek == "Gamma") {return npd1 / (S0_ * volatility_ * sqrtTime); } // Gamma is the same for calls and puts
    else if (greek == "Theta") {
        if (optionType_ == "Call") {
            // Theta for call
            double term1 = - (S0_ * npd1 * volatility_) / (2.0 * sqrtTime);
            double term2 = - riskFreeRate_[0] * strike_ * std::exp(-riskFreeRate_[0] * timeToMaturity) * Nd2;
            return (term1 + term2) / 365.0; // Convert to per day
        } 
        else { // Put
            // Theta for put
            double term1 = - (S0_ * npd1 * volatility_) / (2.0 * sqrtTime);
            double term2 = riskFreeRate_[0] * strike_ * std::exp(-riskFreeRate_[0] * timeToMaturity) * N_minus_d2;
            return (term1 + term2) / 365.0; // Convert to per day
        }
    } 
    else if (greek == "Vega") {return (S0_ * npd1 * sqrtTime) / 100.0;} // Per 1% change (Vega is the same for calls and puts) 
    else if (greek == "Rho") {
        if (optionType_ == "Call") {
            // Rho for call
            return (strike_ * timeToMaturity * std::exp(-riskFreeRate_[0] * timeToMaturity) * Nd2) / 100.0; // Per 1% change
        } 
        else { // Put
            // Rho for put
            return (-strike_ * timeToMaturity * std::exp(-riskFreeRate_[0] * timeToMaturity) * N_minus_d2) / 100.0; // Per 1% change
        }
    } 
    else {
        throw std::invalid_argument("Invalid Greek requested. Available Greeks: Delta, Gamma, Theta, Vega, Rho.");
    }
}

// Method to numerically compute a greek that is provided as input
double OptionPricer::computeGreekNumerical(const std::string greek) {
    // Finite difference increments
    double h = S0_ * 0.01;       // 1% change in spot price
    double deltaT = 0.01 / 365;  // Small change in time (in days)
    double deltaSigma = 0.01;    // 1% change in volatility
    double deltaR = 0.0001;      // 0.01% change in risk-free rate

    if (greek == "Delta") {
        double price_up = computePricePDE(S0_ + h, maturity_, riskFreeRate_, volatility_);
        double price_down = computePricePDE(S0_ - h, maturity_, riskFreeRate_, volatility_);
        return (price_up - price_down) / (2 * h);
    } 
    else if (greek == "Gamma") {
        double price_up = computePricePDE(S0_ + h, maturity_, riskFreeRate_, volatility_);
        double price_mid = computePricePDE(S0_, maturity_, riskFreeRate_, volatility_);
        double price_down = computePricePDE(S0_ - h, maturity_, riskFreeRate_, volatility_);
        return (price_up - 2 * price_mid + price_down) / (h * h);
    } 
    else if (greek == "Theta") {
        // Ensure time to maturity remains positive
        double T_up = maturity_ - deltaT;
        if (T_up <= T0_) {
            throw std::logic_error("Delta T is too large, time to maturity becomes negative.");
        }
        double price_now = computePricePDE(S0_, maturity_, riskFreeRate_, volatility_);
        double price_later = computePricePDE(S0_, T_up, riskFreeRate_, volatility_);
        return (price_later - price_now) / deltaT / 365.0; // Negative value as time decreases
    } 
    else if (greek == "Vega") {
        double price_up = computePricePDE(S0_, maturity_, riskFreeRate_, volatility_ + deltaSigma);
        double price_down = computePricePDE(S0_, maturity_, riskFreeRate_, volatility_ - deltaSigma);
        return (price_up - price_down) / (2 * deltaSigma) / 100;
    } 
    else if (greek == "Rho") {
        // Compute the vector of the changed risk free rates
        std::vector<double> riskFreeRateUp; 
        std::vector<double> riskFreeRateDown;
        for (int j = 0; j < k_; j++) {
            riskFreeRateUp.push_back(riskFreeRate_[j] + deltaR);
            riskFreeRateDown.push_back(riskFreeRate_[j] - deltaR);
        } 
        double price_up = computePricePDE(S0_, maturity_, riskFreeRateUp, volatility_);
        double price_down = computePricePDE(S0_, maturity_, riskFreeRateDown, volatility_);
        return (price_up - price_down) / (2 * deltaR) / 100;
    } 
    else {
        throw std::invalid_argument("Invalid Greek requested. Available Greeks: Delta, Gamma, Theta, Vega, Rho.");
    }
    return 0.0;
}

// Actual function that the user can use
double OptionPricer::computeGreek(const std::string greek, const std::string method) {
    if (method == "Black-Scholes") {
        return computeGreekBS(greek);
    }
    else if (method == "PDE") {
        return computeGreekNumerical(greek);
    }
    else {
        throw std::invalid_argument("Method for computing option greek must be either 'Black-Scholes' or 'PDE'.");
    }
}

// Function to print the comparison of prices
void OptionPricer::compareGreeks(std::string greek) {
    double PDE_greek = computeGreekNumerical(greek);
    double BS_greek = computeGreekBS(greek);

    double PDE_error = PDE_greek - BS_greek;

    std::cout << "Black-Scholes " << greek << ": " << BS_greek << ". Finite-Differences " << greek << ": " << PDE_greek << ". PDE Error: " << PDE_error << ". PDE Relative Error: " << std::abs(PDE_error / BS_greek) * 100 << "%." << std::endl;
}

} // namespace project
