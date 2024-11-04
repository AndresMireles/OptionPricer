// option.cpp

#include "option.h"
#include <string>
#include <stdexcept>

namespace project {

// Default constructor with default values
Option::Option() {
    optionType_ = "Call";
    exerciseType_ = "European";
    strike_ = 0.0;
    maturity_ = 0.0;
    volatility_ = 0.0;
}


// Constructor
Option::Option(
    const std::string& optionType, 
    const std::string& exerciseType, 
    double strike, 
    double maturity, 
    double volatility
) {
    if (optionType != "Call" && optionType != "Put") {
        throw std::invalid_argument("Option type must be either 'call' or 'put'.");
    }   
    if (exerciseType != "European" && exerciseType != "American") {
        throw std::invalid_argument("Exercise type must be either 'European' or 'American'.");
    } 
    if (strike < 0) {
        throw std::invalid_argument("Strike must be greater than 0.");
    }  
    if (maturity < 0) {
        throw std::invalid_argument("Maturity must be greater than 0.");
    }    
    if (volatility < 0) {
        throw std::invalid_argument("Volatility price must be greater than 0.");
    }    

    // Set the elements to the private variables
    optionType_ = optionType;
    exerciseType_ = exerciseType;
    strike_ = strike;
    maturity_ = maturity;
    volatility_ = volatility;
}


// Overload of << operator
std::ostream& operator<<(std::ostream& os, const project::Option& opt) {
    os << "Option Type: " << opt.getOptionType() << std::endl;
    os << "Exercise Type: " << opt.getExerciseType() << std::endl;
    os << "Strike: " << opt.getStrike() << " $" << std::endl;
    os << "Maturity (years): " << opt.getMaturity() << std::endl;
    os << "Annualized Volatility: " << opt.getVolatility() * 100 << " %" << std::endl;

    return os;
}

} // namespace project
