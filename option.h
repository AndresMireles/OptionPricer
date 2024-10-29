// option.h

#ifndef OPTION_H
#define OPTION_H

#include <string>
#include <iostream>

namespace project {

class Option {

private:
    // Option parameters
    std::string optionType_;
    std::string exerciseType_;
    double strike_;
    double maturity_;
    double volatility_;

public:
    // Constructor
    Option(
        const std::string& optionType,
        const std::string& exerciseType,
        double strike,
        double maturity,
        double volatility
    );

    // Getters
    std::string getOptionType() const { return optionType_; }
    std::string getExerciseType() const { return exerciseType_; }
    double getStrike() const { return strike_; }
    double getMaturity() const { return maturity_; }
    double getVolatility() const { return volatility_; }

};

// Overload << operator
std::ostream& operator<<(std::ostream & st, const Option& opt);

} // namespace project

#endif // OPTION_H
