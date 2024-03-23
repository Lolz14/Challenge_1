#include "gd.h"



/* Example usage
Call make in the command line. The compiling time is quite long, since exprtk is an expensive library, all-in-one header.
In order to use the program, it's necessary to call the main and add at command line a function, which can be as follows:
./main "y^2-x*3+x*y" n
where n is the algorithm chosen to adjust the learning steprate.
The whole project puts its focus on the concept of functors, it uses exprtk as a (working on my pc:() alternative to muparser)
Moreover Eigen is the library that was used to take advantage of its efficiency*/


int main(int argc, char** argv) {
    // Define the objective function

    std::string expression_string = argv[1];
    std::vector<double> variables = { 1.0, 2.0 }; // Example variables


    Function<double> objective(expression_string, variables);
 



    if (std::stoi(argv[2])==0){
      GradientDescent<double,CentralDifferences<double>,ArmijoLearningRate<double>> optimizer;
      Eigen::VectorXd initialGuess(2);
      initialGuess << 0.5, 0.5; // Initial guess

      Eigen::VectorXd result = optimizer.minimize(objective, initialGuess);
          std::cout << "Optimal solution: " << result.transpose() << std::endl;


    }
    else if (std::stoi(argv[2])==1){
         GradientDescent<double,CentralDifferences<double>,ExponentialDecay<double>> optimizer;
      Eigen::VectorXd initialGuess(2);
      initialGuess << 0.5, 0.5; // Initial guess

      Eigen::VectorXd result = optimizer.minimize(objective, initialGuess);
          std::cout << "Optimal solution: " << result.transpose() << std::endl;


    }else
    {
      GradientDescent<double,CentralDifferences<double>,InverseDecay<double>> optimizer;
      Eigen::VectorXd initialGuess(2);
      initialGuess << 0.5, 0.5; // Initial guess

      Eigen::VectorXd result = optimizer.minimize(objective, initialGuess);
          std::cout << "Optimal solution: " << result.transpose() << std::endl;

    }
    
    


          
    return 0;
}




