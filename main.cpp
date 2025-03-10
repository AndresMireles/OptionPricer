#include "option.h"
#include "optionpricer.h"
#include <iostream>
#include <string>
#include <typeinfo>
#include <fstream>
#include <sstream>
#include <map>
#include <stdexcept>
#include <chrono>
#include <cmath>

void computeTimesnk() {
    std::ofstream outFile("pricing_errors.csv");
    
    // Write CSV header
    outFile << "n,k,Relative PDE Error,Time Taken\n";
    
    // Loop over n and k (in multiples of 10 from 10 to 10000)
    for (double log_n = 1; log_n <= 4; log_n += 0.5) {
        for (double log_k = 1; log_k <= 4; log_k += 0.5) {

            int n = floor(pow(10, log_n));
            int k = floor(pow(10, log_k));

            // Record the start time
            auto start = std::chrono::high_resolution_clock::now();
            
            // Create Option object
            project::Option option("Call", "European", 100*exp(0.05), 1.0, 0.2);
            
            // Create OptionPricer object with the current n and k values
            std::vector<double> riskFreeTimes = {0.0, 1};
            std::vector<double> riskFreeRates = {0.05, 0.05};
            project::OptionPricer pricer(option, n, k, 100.0, riskFreeTimes, riskFreeRates, 0.0, 0.0);
            
            // Compute the price using PDE method
            double pdePrice = pricer.computePrice("PDE");
            
            // Compute the price using Black-Scholes method
            double bsPrice = pricer.computePrice("Black-Scholes");
            
            // Compute the error between PDE and Black-Scholes
            double error = std::abs(pdePrice - bsPrice) / bsPrice;
            
            // Record the end time
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end - start;
            
            // Write the results to the CSV
            outFile << n << "," << k << "," << error << "," << duration.count() << "\n";
            
            // Print progress to console
            std::cout << "n: " << n << ", k: " << k << " - Error: " << error << ", Time Taken: " << duration.count() << "s\n";
        }
    }
    
    outFile.close();
    std::cout << "Test completed. Results saved to pricing_errors.csv." << std::endl;
}


int main() {

    // computeTimesnk();

    // Read the input csv file
    std::ifstream file("C:\\Users\\andre\\OneDrive\\Escritorio\\M2QF Paris-Saclay\\Asignaturas\\1er Cuatri\\Project Info\\OptionPricer\\value.csv");
    std::string line;
    std::map<std::string, std::string> variables;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string variable_name, variable_value;
        if (std::getline(ss, variable_name, ',') && std::getline(ss, variable_value, ',')) {
            variables[variable_name] = variable_value;
            // std::cout << variable_name << " " << variable_value << std::endl;
        }
    }

    file.close();

    // Assign the values of the variables
    std::string optionType = variables["Contract Type"];
    std::string exerciseType = variables["Exercise Type"];
    double maturity = std::stod(variables["Maturity T"]);
    double strike = std::stod(variables["Strike"]);
    double T0 = std::stod(variables["T0"]);
    double S0 = std::stod(variables["S0"]);
    double volatility = std::stod(variables["Volatility"]);
    int n = std::stoi(variables["n"]);
    int k = std::stoi(variables["k"]);
    double q = std::stod(variables["q"]);
    double r_T0 = std::stod(variables["r_T0"]);
    double r_T = std::stod(variables["r_T"]);
    double r_T0_P = std::stod(variables["r_T0_P"]);
    double r_T_P = std::stod(variables["r_T_P"]);
    double r_T0_N = std::stod(variables["r_T0_N"]);
    double r_T_N = std::stod(variables["r_T_N"]);
    std::string range_param = variables["range_param"];
    double range_min = std::stod(variables["range_min"]);
    double range_max = std::stod(variables["range_max"]);
    int n_points = std::stoi(variables["n_points"]);
    std::string greek = variables["greek"];

    // Build the risk free rates vectors
    std::vector<double> riskFreeTimes = {0.0, maturity};
    std::vector<double> riskFreeRates = {r_T0,r_T};
    std::vector<double> riskFreeRates_P = {r_T0_P,r_T_P};
    std::vector<double> riskFreeRates_N = {r_T0_N,r_T_N};

    // Build the option objects for all the options that we need
    project::Option opt(optionType, exerciseType, strike, maturity, volatility);
    project::Option optAm(optionType, "American", strike, maturity, volatility);
    project::Option opt1("Call", "European", strike, maturity, volatility);
    project::Option opt2("Call", "American", strike, maturity, volatility);
    project::Option opt3("Put", "European", strike, maturity, volatility);
    project::Option opt4("Put", "American", strike, maturity, volatility);
    
    // Build the pricers for each option type
    project::OptionPricer pricer(opt, n, k, S0, riskFreeTimes, riskFreeRates, q, T0);
    project::OptionPricer pricerAm(optAm, n, k, S0, riskFreeTimes, riskFreeRates, q, T0);

    project::OptionPricer pricer_p1(opt1, n, k, S0, riskFreeTimes, riskFreeRates_P, q, T0);
    project::OptionPricer pricer_p2(opt2, n, k, S0, riskFreeTimes, riskFreeRates_P, q, T0);
    project::OptionPricer pricer_p3(opt3, n, k, S0, riskFreeTimes, riskFreeRates_P, q, T0);
    project::OptionPricer pricer_p4(opt4, n, k, S0, riskFreeTimes, riskFreeRates_P, q, T0);

    project::OptionPricer pricer_n1(opt1, n, k, S0, riskFreeTimes, riskFreeRates_N, q, T0);
    project::OptionPricer pricer_n2(opt2, n, k, S0, riskFreeTimes, riskFreeRates_N, q, T0);
    project::OptionPricer pricer_n3(opt3, n, k, S0, riskFreeTimes, riskFreeRates_N, q, T0);
    project::OptionPricer pricer_n4(opt4, n, k, S0, riskFreeTimes, riskFreeRates_N, q, T0);

    // Compute Black-Scholes prices
    std::cout << std::endl;
    std::cout << "Computing Black-Scholes prices and greeks ..." << std::endl << std::endl;
    double BS_price;
    double BS_price_delta;
    double BS_price_gamma;
    double BS_price_theta;
    double BS_price_vega;
    double BS_price_rho;
    if (exerciseType == "European") {
        BS_price = pricer.computePrice("Black-Scholes");
        BS_price_delta = pricer.computeGreek("Delta","Black-Scholes");
        BS_price_gamma = pricer.computeGreek("Gamma","Black-Scholes");
        BS_price_theta = pricer.computeGreek("Theta","Black-Scholes");
        BS_price_vega = pricer.computeGreek("Vega","Black-Scholes");
        BS_price_rho = pricer.computeGreek("Rho","Black-Scholes");
    }
    else {
        BS_price = std::nan("");
        BS_price_delta = std::nan("");
        BS_price_gamma = std::nan("");
        BS_price_theta = std::nan("");
        BS_price_vega = std::nan("");
        BS_price_rho = std::nan("");
    }

    // Compute PDE prices
    std::cout << "Computing Numerical prices and greeks ..." << std::endl << std::endl;
    double PDE_price = pricer.computePrice("PDE");
    double num_delta = pricer.computeGreek("Delta","Numerical");
    double num_gamma = pricer.computeGreek("Gamma","Numerical");
    double num_theta = pricer.computeGreek("Theta","Numerical");
    double num_vega = pricer.computeGreek("Vega","Numerical");
    double num_rho = pricer.computeGreek("Rho","Numerical");

    // Compute the prices of the options to check the equivalence between European and American options
    std::cout << "Computing PDE (European-American equivalence) prices ..." << std::endl << std::endl;
    double PDE_price_p1= pricer_p1.computePrice("PDE");
    double PDE_price_p2= pricer_p2.computePrice("PDE");
    double PDE_price_p3= pricer_p3.computePrice("PDE");
    double PDE_price_p4= pricer_p4.computePrice("PDE");

    double PDE_price_n1= pricer_n1.computePrice("PDE");
    double PDE_price_n2= pricer_n2.computePrice("PDE");
    double PDE_price_n3= pricer_n3.computePrice("PDE");
    double PDE_price_n4= pricer_n4.computePrice("PDE");

    // Errors and Relative Errors
    double PDE_error=PDE_price-BS_price;
    double PDE_relative_error=(PDE_price-BS_price)/BS_price;

    double num_error_delta = num_delta - BS_price_delta;
    double num_relative_error_delta = num_error_delta / BS_price_delta;

    double num_error_gamma = num_gamma - BS_price_gamma;
    double num_relative_error_gamma = num_error_gamma / BS_price_gamma;

    double num_error_theta = num_theta - BS_price_theta;
    double num_relative_error_theta = num_error_theta / BS_price_theta;

    double num_error_vega = num_vega - BS_price_vega;
    double num_relative_error_vega = num_error_vega / BS_price_vega;

    double num_error_rho = num_rho - BS_price_rho;
    double num_relative_error_rho = num_error_rho / BS_price_rho;
    
    // Compute the values of the prices and greeks for the given range
    std::vector<double> range;
    double step = (range_max - range_min) / (n_points - 1);

    for (int i = 0; i < n_points; ++i) {
        range.push_back(range_min + i * step);
    }

    std::cout << "Computing prices vector ..." << std::endl << std::endl;
    std::vector<double> prices = pricer.computePricesVector(range_param, range);
    std::cout << "Computing greeks vector ..." << std::endl << std::endl;
    std::vector<double> greeks = pricer.computeGreeksVector(greek, range_param, range);

    // Save results into different files
    std::cout << "Saving output files ..." << std::endl << std::endl;

    // Open an output file to save the results
    std::ofstream prices_greeks_file("C:\\Users\\andre\\OneDrive\\Escritorio\\M2QF Paris-Saclay\\Asignaturas\\1er Cuatri\\Project Info\\OptionPricer\\prices_greeks.csv");
    if (!prices_greeks_file.is_open()) {
        std::cerr << "Error: Unable to open the output file." << std::endl;
        return 1;
    }

    // Write the headers and results to the CSV file
    prices_greeks_file << "Metric,Black-Scholes,Numerical,Absolute Error,Relative Error\n";
    prices_greeks_file << "price," << BS_price << "," << PDE_price << "," << PDE_error << "," << PDE_relative_error << "\n";
    prices_greeks_file << "price_Delta," << BS_price_delta << "," << num_delta << "," << num_error_delta << "," << num_relative_error_delta << "\n";
    prices_greeks_file << "price_Gamma," << BS_price_gamma << "," << num_gamma << "," << num_error_gamma << "," << num_relative_error_gamma << "\n";
    prices_greeks_file << "price_Theta," << BS_price_theta << "," << num_theta << "," << num_error_theta << "," << num_relative_error_theta << "\n";
    prices_greeks_file << "price_Vega," << BS_price_vega << "," << num_vega << "," << num_error_vega << "," << num_relative_error_vega << "\n";
    prices_greeks_file << "price_Rho," << BS_price_rho << "," << num_rho << "," << num_error_rho << "," << num_relative_error_rho << "\n";

    prices_greeks_file.close();

    std::ofstream eur_am_equiv_file("C:\\Users\\andre\\OneDrive\\Escritorio\\M2QF Paris-Saclay\\Asignaturas\\1er Cuatri\\Project Info\\OptionPricer\\eur_am_equiv.csv");
    if (!eur_am_equiv_file.is_open()) {
        std::cerr << "Error: Unable to open the output file." << std::endl;
        return 1;
    }

    //Add PDE prices to the CSV file
    eur_am_equiv_file << "PositiveNegativePrices,Price" "\n";
    eur_am_equiv_file << "PDE_price_P1," << PDE_price_p1 << "\n";
    eur_am_equiv_file << "PDE_price_P2," << PDE_price_p2 << "\n";
    eur_am_equiv_file << "PDE_price_P3," << PDE_price_p3 << "\n";
    eur_am_equiv_file << "PDE_price_P4," << PDE_price_p4 << "\n";
    eur_am_equiv_file << "PDE_price_N1," << PDE_price_n1 << "\n";
    eur_am_equiv_file << "PDE_price_N2," << PDE_price_n2 << "\n";
    eur_am_equiv_file << "PDE_price_N3," << PDE_price_n3 << "\n";
    eur_am_equiv_file << "PDE_price_N4," << PDE_price_n4 << "\n";

    eur_am_equiv_file.close();

    std::ofstream range_file("C:\\Users\\andre\\OneDrive\\Escritorio\\M2QF Paris-Saclay\\Asignaturas\\1er Cuatri\\Project Info\\OptionPricer\\ranges.csv");
    if (!range_file.is_open()) {
        std::cerr << "Error: Unable to open the output file." << std::endl;
        return 1;
    }

    //Add PDE prices to the CSV file
    range_file << "ParamValue,Price,Greek" "\n";
    for (int i = 0; i < n_points; ++i) {
        range_file << range[i] << "," << prices[i] << "," << greeks[i] << "\n"; 
    }

    range_file.close();
        
    // Also save the exercise boundary of an American option
    pricerAm.saveExerciseBoundaryToFile("C:\\Users\\andre\\OneDrive\\Escritorio\\M2QF Paris-Saclay\\Asignaturas\\1er Cuatri\\Project Info\\OptionPricer\\boundary.csv");

    std::cout << "Process finished";
    
    return 0;
}