#include "option.h"
#include "optionpricer.h"

#include <iostream>
#include <string>

int main() {

    project::Option opt = project::Option("Call", "European", 110.0, 1.5, 0.1);

    std::cout << opt;

    return 0;
}