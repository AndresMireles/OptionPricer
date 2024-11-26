#include "OptionPricer.h"
#include <cmath>
#include <fstream>
namespace project {

const double DAYS_IN_A_YEAR = 365.0;

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

// The method for setting up the grid of prices
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
        } 
        else {
            grid_[i][k_] = std::max(strike_ - spot, 0.0);
        }
    }

    // Boundary condition for S = 0
    for (int j = 0; j <= k_; ++j) {
        double t = timeSteps_[j];
        double r = interpolateRiskFreeRate(t, deltaR);
        if (isCall) {
            grid_[0][j] = 0.0; // Call option is worthless
        } 
        else {
            grid_[0][j] = strike_ * exp(-r * (maturity - t));
        }
    }

    // Boundary condition for very high S (highest spot price)
    for (int j = 0; j <= k_; ++j) {
        double t = timeSteps_[j];
        double r = interpolateRiskFreeRate(t, deltaR);
        if (isCall) {
            grid_[n_][j] = spotPrices_[n_] * exp(-q * (maturity - t)) - strike_ * exp(-r * (maturity - t));
        } 
        else {
            grid_[n_][j] = 0.0; // Put option is worthless
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
        std::vector<double> a(N, 0.0); // Lower diagonal
        std::vector<double> b(N, 0.0); // Main diagonal
        std::vector<double> c(N, 0.0); // Upper diagonal
        std::vector<double> d(N, 0.0); // Right-hand side

        double t = timeSteps_[j]; 
        double r = interpolateRiskFreeRate(t, deltaR); // Use risk-free rate at current time step (interpolating)

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
            }
            else {
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

    int spotIndex;
    if (n_ % 2 == 0) {
        int lowerMid = n_ / 2 - 1;
        int upperMid = n_ / 2;
        return (grid_[lowerMid][0] + grid_[upperMid][0]) / 2.0;
    } 
    else {
        spotIndex = n_ / 2;
        return grid_[spotIndex][0];
    }
}

// Function to compute the cumulative distribution function for the standard normal distribution
double OptionPricer::normCDF(double x) const {
    return 0.5 * (1.0 + std::erf(x / sqrt(2.0))); // erf function comes from cmath
}

double OptionPricer::computePriceBS() {

    if (exerciseType_ != "European") {
        throw std::logic_error("Black-Scholes price is only applicable to European options.");
    }

    double timeToMaturity = maturity_ - T0_;
    double r = interpolateRiskFreeRate(timeToMaturity);
    double q = dividendYield_;

    // Calculate d1 and d2
    double d1 = (log(S0_ / strike_) + (r - q + 0.5 * volatility_ * volatility_) * timeToMaturity) / (volatility_ * sqrt(timeToMaturity));
    double d2 = d1 - volatility_ * sqrt(timeToMaturity);

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
    double PDEPrice = computePricePDE(S0_, maturity_, volatility_);
    double BSPrice = computePriceBS();

    double PDEError = PDEPrice - BSPrice;

    std::cout << "Black-Scholes price: " << BSPrice << ". PDE Price: " << PDEPrice << ". PDE Error: " << PDEError << ". PDE Relative Error: " << std::abs(PDEError / BSPrice) * 100 << "%." << std::endl;
}

// Implementation of the standard normal PDF
double OptionPricer::normPDF(double x) const {
    return (1.0 / sqrt(2.0 * M_PI)) * exp(-0.5 * x * x);
}

// Method to analytically cally compute a greek that is provided as input (only valid for European options)
double OptionPricer::computeGreekBS(const std::string greek) {
    if (exerciseType_ != "European") {
        throw std::logic_error("Greeks can only be computed analytically for European options.");
    }

    double timeToMaturity = maturity_ - T0_;
    double r = interpolateRiskFreeRate(T0_);
    double q = dividendYield_;
    double sqrtTime = sqrt(timeToMaturity);

    // Calculate d1 and d2 with dividend yield
    double d1 = (log(S0_ / strike_) + (r - q + 0.5 * volatility_ * volatility_) * timeToMaturity) /
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
            return (term1 + term2 + term3) / DAYS_IN_A_YEAR; // Convert to per day
        } 
        else {
            double term1 = - (S0_ * npd1 * volatility_ * exp(-q * timeToMaturity)) / (2.0 * sqrtTime);
            double term2 = - q * S0_ * exp(-q * timeToMaturity) * N_minus_d1;
            double term3 = r * strike_ * exp(-r * timeToMaturity) * N_minus_d2;
            return (term1 + term2 + term3) / DAYS_IN_A_YEAR; // Convert to per day
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
double OptionPricer::computeGreekNumerical(const std::string greek, double S0, double maturity, double volatility) {
    // Finite difference increments
    double h = 0.01; // 1% change in spot price
    double deltaT = 1.0 / DAYS_IN_A_YEAR; // One day change in maturity (in years)
    double deltaVolatility = 0.01; // 1% change in volatility
    double deltaR = 0.0001; // 0.01% change in risk-free rate

    if (greek == "Delta") {
        double priceUp = computePricePDE(S0 * (1 + h), maturity, volatility);
        double priceDown = computePricePDE(S0 * (1 - h), maturity, volatility);
        return (priceUp - priceDown) / (2 * S0 * h);
    } 
    else if (greek == "Gamma") {
        double priceUp = computePricePDE(S0 * (1 + h), maturity, volatility);
        double price_mid = computePricePDE(S0, maturity, volatility);
        double priceDown = computePricePDE(S0 * (1 - h), maturity, volatility);
        return (priceUp - 2 * price_mid + priceDown) / ((S0 * h) * (S0 * h));
    } 
    else if (greek == "Theta") {
        // Ensure time to maturity remains positive
        double T_up = maturity - deltaT;
        if (T_up <= T0_) {
            throw std::logic_error("Delta T is too large, time to maturity becomes negative.");
        }
        double price_now = computePricePDE(S0, maturity, volatility);
        double price_later = computePricePDE(S0, T_up, volatility);
        return (price_later - price_now) / deltaT / DAYS_IN_A_YEAR; // Negative value as time decreases
    } 
    else if (greek == "Vega") {
        double priceUp = computePricePDE(S0, maturity, volatility + deltaVolatility);
        double priceDown = computePricePDE(S0, maturity, volatility - deltaVolatility);
        return (priceUp - priceDown) / (2 * deltaVolatility) / 100;
    } 
    else if (greek == "Rho") {
        // Compute the price with increased and decreased risk-free rates
        double priceUp = computePricePDE(S0, maturity, volatility, deltaR);
        double priceDown = computePricePDE(S0, maturity, volatility, -deltaR);
        return (priceUp - priceDown) / (2 * deltaR) / 100; // Per 1% change
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
        return computeGreekNumerical(greek, S0_, maturity_, volatility_);
    }
    else {
        throw std::invalid_argument("Method for computing option greek must be either 'Black-Scholes' or 'Numerical'.");
    }
}

// Function to print the comparison of prices
void OptionPricer::compareGreeks(std::string greek) {
    double numericalGreek = computeGreekNumerical(greek, S0_, maturity_, volatility_);
    double BSGreek = computeGreekBS(greek);

    double numericalError = numericalGreek - BSGreek;

    std::cout << "Black-Scholes " << greek << ": " << BSGreek << ". Finite-Differences " << greek << ": " << numericalGreek << ". Numerical Error: " << numericalError << ". Numerical Relative Error: " << std::abs(numericalError / BSGreek) * 100 << "%." << std::endl;
}

// Method to compute PDE prices for a range of values
std::vector<double> OptionPricer::computePricesVector(const std::string param, const std::vector<double> paramRange) {    
    if (param != "Spot" && param != "Maturity" && param != "Volatility") {
        throw std::invalid_argument("Param name should be 'Spot', 'Maturity' or 'Volatility'");
    }

    std::vector<double> prices(paramRange.size());

    for (unsigned int i = 0; i < paramRange.size(); i++) {
        double v = paramRange[i];
        if (param == "Spot") {
            prices[i] = computePricePDE(v, maturity_, volatility_);
        }
        else if (param == "Maturity") {
            prices[i] = computePricePDE(S0_, v, volatility_);
        }
        else if (param == "Volatility") {
            prices[i] = computePricePDE(S0_, maturity_, v);
        }
    }

    return prices;
}

// Method to compute PDE greeks for a range of values
std::vector<double> OptionPricer::computeGreeksVector(const std::string greek, const std::string param, const std::vector<double> paramRange) {
    if (param != "Spot" && param != "Maturity" && param != "Volatility") {
        throw std::invalid_argument("Param name should be 'Spot', 'Maturity' or 'Volatility'");
    }

    std::vector<double> greeks(paramRange.size());

    for (unsigned int i = 0; i < paramRange.size(); i++) {
        double v = paramRange[i];
        if (param == "Spot") {
            greeks[i] = computeGreekNumerical(greek, v, maturity_, volatility_);
        }
        else if (param == "Maturity") {
            greeks[i] = computeGreekNumerical(greek, S0_, v, volatility_);
        }
        else if (param == "Volatility") {
            greeks[i] = computeGreekNumerical(greek, S0_, maturity_, v);
        }
    }

    return greeks;
}

std::vector<std::pair<double, double>> OptionPricer::computeExerciseBoundary() {

    // We compute the price using the parameters of the pricer (to avoid errors if other methods are executed that modify the grid)
    computePricePDE(S0_, maturity_, volatility_);

    std::vector<std::pair<double, double>> exerciseBoundary;
    double tolerance = 1e-6; // Tolerance for floating-point comparison

    // Loop over time steps
    for (int j = 0; j <= k_; ++j) {
        double timeToMaturity = maturity_ - timeSteps_[j];
        double boundaryPrice = -1.0;

        // For an American Put Option
        if (optionType_ == "Put" && exerciseType_ == "American") {
            // Loop over spot prices
            for (int i = 0; i <= n_; ++i) {
                double spotPrice = spotPrices_[i];
                double optionValue = grid_[i][j];
                double payoff = std::max(strike_ - spotPrice, 0.0);

                // Check if the option value equals the payoff within tolerance
                if (std::abs(optionValue - payoff) < tolerance) {
                    boundaryPrice = spotPrice;
                    break; // Found the boundary
                }
            }
        }
        // For an American Call Option
        else if (optionType_ == "Call" && exerciseType_ == "American") {
            // Loop over spot prices in reverse
            for (int i = n_; i >= 0; --i) {
                double spotPrice = spotPrices_[i];
                double optionValue = grid_[i][j];
                double payoff = std::max(spotPrice - strike_, 0.0);

                if (std::abs(optionValue - payoff) < tolerance) {
                    boundaryPrice = spotPrice;
                    break;
                }
            }
        }

        // If boundary price was found, add it to the vector
        if (boundaryPrice >= 0.0) {
            exerciseBoundary.push_back(std::make_pair(timeToMaturity, boundaryPrice / strike_));
        }
    }

    return exerciseBoundary;
}

void OptionPricer::saveExerciseBoundaryToFile(const std::string& filename) {
    auto boundary = computeExerciseBoundary();
    std::ofstream file(filename);
    if (file.is_open()) {
        file << "TimeToMaturity,AssetPriceOverStrike\n";
        for (const auto& point : boundary) {
            file << point.first << "," << point.second << "\n";
        }
        file.close();
    }
}


} // namespace project
