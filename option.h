#ifndef OPTION_H
#define OPTION_H

#include <string>
#include <iostream>

namespace project {

/**
 * @class Option
 * @brief Represents a financial option with specific parameters.
 *
 * The Option class encapsulates the fundamental properties of a financial option,
 * including its type (Call or Put), exercise type (European or American), strike price,
 * maturity date, and volatility. It provides constructors for initializing these parameters
 * and getter methods to access their values.
 */
class Option {

private:
    // Option parameters
    std::string optionType_;     ///< Type of the option ("Call" or "Put")
    std::string exerciseType_;   ///< Exercise type of the option ("European" or "American")
    double strike_;              ///< Strike price of the option
    double maturity_;            ///< Maturity date of the option (in years)
    double volatility_;          ///< Volatility of the underlying asset

public:

    /**
     * @brief Default constructor.
     *
     * Initializes an Option object with default values:
     * - optionType_: "Call"
     * - exerciseType_: "European"
     * - strike_: 100.0
     * - maturity_: 1.0 year
     * - volatility_: 0.2 (20%)
     */
    Option();

    /**
     * @brief Parameterized constructor.
     *
     * Constructs an Option object with specified parameters.
     *
     * @param optionType The type of the option ("Call" or "Put").
     * @param exerciseType The exercise type of the option ("European" or "American").
     * @param strike The strike price of the option.
     * @param maturity The maturity date of the option (in years).
     * @param volatility The volatility of the underlying asset.
     *
     * @throws std::invalid_argument if any of the parameters are invalid (e.g., negative strike price).
     */
    Option(
        const std::string& optionType,
        const std::string& exerciseType,
        double strike,
        double maturity,  // Maturity date (in years)
        double volatility
    );

     /**
     * @brief Retrieves the type of the option.
     *
     * @return A string representing the option type ("Call" or "Put").
     */
    std::string getOptionType() const { return optionType_; }

    /**
     * @brief Retrieves the exercise type of the option.
     *
     * @return A string representing the exercise type ("European" or "American").
     */
    std::string getExerciseType() const { return exerciseType_; }

    /**
     * @brief Retrieves the strike price of the option.
     *
     * @return The strike price as a double.
     */
    double getStrike() const { return strike_; }

    /**
     * @brief Retrieves the maturity date of the option.
     *
     * @return The maturity date in years as a double.
     */
    double getMaturity() const { return maturity_; }

    /**
     * @brief Retrieves the volatility of the underlying asset.
     *
     * @return The volatility as a double.
     */
    double getVolatility() const { return volatility_; }

};

/**
 * @brief Overloads the << operator to output the details of an Option object.
 *
 * This function allows you to use the << operator with std::ostream to print
 * the details of an Option object in a readable format.
 *
 * @param st The output stream.
 * @param opt The Option object to be printed.
 * @return A reference to the output stream.
 */
std::ostream& operator<<(std::ostream & st, const Option& opt);

} // namespace project

#endif // OPTION_H
