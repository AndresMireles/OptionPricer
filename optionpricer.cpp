#include "OptionPricer.h"
#include <cmath>

namespace project {

// Constructor
OptionPricer::OptionPricer(
    Option option,
    int n,
    int k,
    double S0,
    std::vector<double> riskFreeTimes,
    std::vector<double> riskFreeRates,
    double dividendYield,
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
    if (riskFreeTimes.size() != riskFreeRates.size()) {
        throw std::invalid_argument("Risk-free times and rates vectors must be of the same size.");
    }

    // Pricer parameters
    n_ = n; // Number of spot steps
    k_ = k; // Number of time steps
    S0_ = S0;
    riskFreeTimes_ = riskFreeTimes;
    riskFreeRates_ = riskFreeRates;
    dividendYield_ = dividendYield;
    T0_ = T0;

    // Option parameters
    optionType_ = option.getOptionType();
    exerciseType_ = option.getExerciseType();
    strike_ = option.getStrike();
    maturity_ = option.getMaturity();
    volatility_ = option.getVolatility();

}

// Method to interpolate risk free rates
double OptionPricer::interpolateRiskFreeRate(double t, double deltaR) {
    if (riskFreeTimes_.empty()) {
        throw std::runtime_error("Risk-free times vector is empty.");
    }
    if (t <= riskFreeTimes_.front()) {
        return riskFreeRates_.front() + deltaR;
    }
    if (t >= riskFreeTimes_.back()) {
        return riskFreeRates_.back() + deltaR;
    }
    // Linear interpolation
    for (size_t i = 0; i < riskFreeTimes_.size() - 1; ++i) {
        if (t >= riskFreeTimes_[i] && t <= riskFreeTimes_[i + 1]) {
            double t1 = riskFreeTimes_[i];
            double t2 = riskFreeTimes_[i + 1];
            double r1 = riskFreeRates_[i];
            double r2 = riskFreeRates_[i + 1];
            double interpolatedRate = r1 + (r2 - r1) * (t - t1) / (t2 - t1);
            return interpolatedRate + deltaR;
        }
    }
    throw std::runtime_error("Failed to interpolate risk-free rate.");
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
void OptionPricer::initializeConditions(double maturity, double deltaR) {
    bool isCall = (optionType_ == "Call");
    double q = dividendYield_;

    // Initial condition at maturity (t = T)
    for (int i = 0; i <= n_; ++i) {
        double spot = spotPrices_[i];
        if (isCall) {
            grid_[i][k_] = std::max(spot - strike_, 0.0);
        } else {
            grid_[i][k_] = std::max(strike_ - spot, 0.0);
        }
    }

    // Boundary condition for S = 0 (lowest spot price)
    for (int j = 0; j <= k_; ++j) {
        double t = timeSteps_[j];
        double r = interpolateRiskFreeRate(t, deltaR);
        if (isCall) {
            grid_[0][j] = 0.0;
        } else {
            grid_[0][j] = strike_ * exp(-r * (maturity - t));
        }
    }

    // Boundary condition for very high S (highest spot price)
    for (int j = 0; j <= k_; ++j) {
        double t = timeSteps_[j];
        double r = interpolateRiskFreeRate(t, deltaR);
        if (isCall) {
            grid_[n_][j] = spotPrices_[n_] * exp(-q * (maturity - t)) - strike_ * exp(-r * (maturity - t));
        } else {
            grid_[n_][j] = 0.0;
        }
    }
}


// This function is general to calls and puts
void OptionPricer::performCalculations(double volatility, double deltaR) {
    // Time and spot step sizes (already computed in setupGrid)
    double dt = timeSteps_[1] - timeSteps_[0];
    double dS = spotPrices_[1] - spotPrices_[0];

    bool isAmerican = (exerciseType_ == "American");
    bool isCall = (optionType_ == "Call");

    double q = dividendYield_;

    // Loop backward in time from maturity to current time T0
    for (int j = k_ - 1; j >= 0; --j) {
        // Set up the tridiagonal system
        int N = n_ - 1; // Number of interior spot points
        std::vector<double> a(N, 0.0); // Lower diagonal (a_1 to a_N)
        std::vector<double> b(N, 0.0); // Main diagonal (b_1 to b_N)
        std::vector<double> c(N, 0.0); // Upper diagonal (c_1 to c_N)
        std::vector<double> d(N, 0.0); // Right-hand side

        double t = timeSteps_[j]; 
        double r = interpolateRiskFreeRate(t, deltaR);; // Use risk-free rate at current time step (interpolating)

        // Loop over interior spot indices
        for (int i = 1; i < n_; ++i) {
            double S = spotPrices_[i];

            // Coefficients for the implicit method
            double alpha = 0.5 * volatility * volatility * S * S / (dS * dS);
            double beta = (r - q) * S / (2.0 * dS);
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
double OptionPricer::computePricePDE(double S0, double maturity, double volatility, double deltaR) {
    setupGrid(S0, maturity);
    initializeConditions(maturity, deltaR);
    performCalculations(volatility, deltaR);

    int spot_index;
    if (n_ % 2 == 0) {
        int lower_mid = n_ / 2 - 1;
        int upper_mid = n_ / 2;
        return (grid_[lower_mid][0] + grid_[upper_mid][0]) / 2.0;
    } else {
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
    double r = interpolateRiskFreeRate(timeToMaturity);
    double q = dividendYield_;

    // Calculate d1 and d2
    double d1 = (std::log(S0_ / strike_) + (r - q + 0.5 * volatility_ * volatility_) * timeToMaturity) / (volatility_ * std::sqrt(timeToMaturity));
    double d2 = d1 - volatility_ * std::sqrt(timeToMaturity);

    // Calculate N(d1) and N(d2)
    double Nd1 = normCDF(d1);
    double Nd2 = normCDF(d2);
    double N_minus_d1 = normCDF(-d1);
    double N_minus_d2 = normCDF(-d2);

    // Compute option price based on type
    double price;
    if (optionType_ == "Call") {
        price = S0_ * exp(-q * timeToMaturity) * Nd1 - strike_ * exp(-r * timeToMaturity) * Nd2;
    } 
    else { // Put
        price = strike_ * exp(-r * timeToMaturity) * N_minus_d2 - S0_ * exp(-q * timeToMaturity) * N_minus_d1;
    }

    return price;
}

// Actual function that the user can use
double OptionPricer::computePrice(const std::string method) {
    if (method == "Black-Scholes") {
        return computePriceBS();
    }
    else if (method == "PDE") {
        return computePricePDE(S0_, maturity_, volatility_);
    }
    else {
        throw std::invalid_argument("Method for computing option price must be either 'Black-Scholes' or 'PDE'.");
    }
}

// Function to print the comparison of prices
void OptionPricer::comparePrices() {
    double PDE_price = computePricePDE(S0_, maturity_, volatility_);
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
    if (exerciseType_ != "European") {
        throw std::logic_error("Greeks can only be computed analytically for European options.");
    }

    double timeToMaturity = maturity_ - T0_;
    double r = interpolateRiskFreeRate(T0_);
    double q = dividendYield_;
    double sqrtTime = std::sqrt(timeToMaturity);

    // Calculate d1 and d2 with dividend yield
    double d1 = (std::log(S0_ / strike_) + (r - q + 0.5 * volatility_ * volatility_) * timeToMaturity) /
                (volatility_ * sqrtTime);
    double d2 = d1 - volatility_ * sqrtTime;

    // Calculate N(d1), N(d2), N(-d2), and N'(d1)
    double Nd1 = normCDF(d1);
    double Nd2 = normCDF(d2);
    double N_minus_d1 = normCDF(-d1);
    double N_minus_d2 = normCDF(-d2);
    double npd1 = normPDF(d1);

    // Compute Greeks with dividend yield
    if (greek == "Delta") {
        if (optionType_ == "Call") {return exp(-q * timeToMaturity) * Nd1;} // Call delta 
        else {return exp(-q * timeToMaturity) * (Nd1 - 1.0);} // Put delta
    } 
    else if (greek == "Gamma") {return (exp(-q * timeToMaturity) * npd1) / (S0_ * volatility_ * sqrtTime);} 
    else if (greek == "Theta") {
        if (optionType_ == "Call") {
            double term1 = - (S0_ * npd1 * volatility_ * exp(-q * timeToMaturity)) / (2.0 * sqrtTime);
            double term2 = q * S0_ * exp(-q * timeToMaturity) * Nd1;
            double term3 = - r * strike_ * exp(-r * timeToMaturity) * Nd2;
            return (term1 + term2 + term3) / 365.0; // Convert to per day
        } 
        else {
            double term1 = - (S0_ * npd1 * volatility_ * exp(-q * timeToMaturity)) / (2.0 * sqrtTime);
            double term2 = - q * S0_ * exp(-q * timeToMaturity) * N_minus_d1;
            double term3 = r * strike_ * exp(-r * timeToMaturity) * N_minus_d2;
            return (term1 + term2 + term3) / 365.0; // Convert to per day
        }
    } 
    else if (greek == "Vega") {return (S0_ * exp(-q * timeToMaturity) * npd1 * sqrtTime) / 100.0;} // Per 1% change 
    else if (greek == "Rho") {
        if (optionType_ == "Call") {return (strike_ * timeToMaturity * exp(-r * timeToMaturity) * Nd2) / 100.0;} // Per 1% change 
        else {return (-strike_ * timeToMaturity * exp(-r * timeToMaturity) * N_minus_d2) / 100.0;} // Per 1% change
    } 
    else {throw std::invalid_argument("Invalid Greek requested. Available Greeks: Delta, Gamma, Theta, Vega and Rho.");}
}

// Method to compute a greek numerically
double OptionPricer::computeGreekNumerical(const std::string greek) {
    // Finite difference increments
    double h = 0.01;             // 1% change in spot price
    double deltaT = 1.0 / 365.0; // One day change in maturity (in years)
    double deltaVolatility = 0.01; // 1% change in volatility
    double deltaR = 0.0001;      // 0.01% change in risk-free rate

    if (greek == "Delta") {
        double price_up = computePricePDE(S0_ * (1 + h), maturity_, volatility_);
        double price_down = computePricePDE(S0_ * (1 - h), maturity_, volatility_);
        return (price_up - price_down) / (2 * S0_ * h);
    } 
    else if (greek == "Gamma") {
        double price_up = computePricePDE(S0_ * (1 + h), maturity_, volatility_);
        double price_mid = computePricePDE(S0_, maturity_, volatility_);
        double price_down = computePricePDE(S0_ * (1 - h), maturity_, volatility_);
        return (price_up - 2 * price_mid + price_down) / ((S0_ * h) * (S0_ * h));
    } 
    else if (greek == "Theta") {
        // Ensure time to maturity remains positive
        double T_up = maturity_ - deltaT;
        if (T_up <= T0_) {
            throw std::logic_error("Delta T is too large, time to maturity becomes negative.");
        }
        double price_now = computePricePDE(S0_, maturity_, volatility_);
        double price_later = computePricePDE(S0_, T_up, volatility_);
        return (price_later - price_now) / deltaT / 365.0; // Negative value as time decreases
    } 
    else if (greek == "Vega") {
        double price_up = computePricePDE(S0_, maturity_, volatility_ + deltaVolatility);
        double price_down = computePricePDE(S0_, maturity_, volatility_ - deltaVolatility);
        return (price_up - price_down) / (2 * deltaVolatility) / 100;
    } 
    else if (greek == "Rho") {
        // Compute the price with increased and decreased risk-free rates
        double price_up = computePricePDE(S0_, maturity_, volatility_, deltaR);
        double price_down = computePricePDE(S0_, maturity_, volatility_, -deltaR);
        return (price_up - price_down) / (2 * deltaR) / 100; // Per 1% change
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
    else if (method == "Numerical") {
        return computeGreekNumerical(greek);
    }
    else {
        throw std::invalid_argument("Method for computing option greek must be either 'Black-Scholes' or 'Numerical'.");
    }
}

// Function to print the comparison of prices
void OptionPricer::compareGreeks(std::string greek) {
    double numerical_greek = computeGreekNumerical(greek);
    double BS_greek = computeGreekBS(greek);

    double numerical_error = numerical_greek - BS_greek;

    std::cout << "Black-Scholes " << greek << ": " << BS_greek << ". Finite-Differences " << greek << ": " << numerical_greek << ". Numerical Error: " << numerical_error << ". Numerical Relative Error: " << std::abs(numerical_error / BS_greek) * 100 << "%." << std::endl;
}

} // namespace project
