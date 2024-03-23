#include "Function.hpp"

/* The GradientDescent algorithm has its roots on Eigen and template structures, to improve flexibility.
Obviously, there are a number of add-ons that one could make, starting from the parameters reading from files, with getpot or json*/

template<typename Scalar>
class CentralDifferences {
public:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

    void operator()(const Function<Scalar>& objective, const Vector& x, Scalar& fval, Vector& gradient) const {
        const Scalar epsilon = 1e-6;
        gradient.resize(x.size());
        for (int i = 0; i < x.size(); ++i) {
            Vector perturbed_x1 = x;
            perturbed_x1[i] += epsilon;
            Vector perturbed_x2 = x;
            perturbed_x2[i] -= epsilon;
            Scalar fval_perturbed1 = objective(perturbed_x1, gradient);
            Scalar fval_perturbed2 = objective(perturbed_x2, gradient);
            gradient[i] = (fval_perturbed1 - fval_perturbed2) / (2 * epsilon);
        }
        fval = objective(x, gradient);
    }
};

template<typename Scalar>
class ForwardDifferences {
public:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

    void operator()(const Function<Scalar>& objective, const Vector& x, Scalar& fval, Vector& gradient) const {
        const Scalar epsilon = 1e-6;
        gradient.resize(x.size());
        for (int i = 0; i < x.size(); ++i) {
            Vector perturbed_x = x;
            perturbed_x[i] += epsilon;
            Scalar fval_perturbed = objective(perturbed_x, gradient);
            gradient[i] = (fval_perturbed - fval) / epsilon;
        }
        fval = objective(x, gradient);
    }
};

template<typename Scalar>
class BackwardDifferences {
public:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

    void operator()(const Function<Scalar>& objective, const Vector& x, Scalar& fval, Vector& gradient) const {
        const Scalar epsilon = 1e-6;
        gradient.resize(x.size());
        for (int i = 0; i < x.size(); ++i) {
            Vector perturbed_x = x;
            perturbed_x[i] -= epsilon;
            Scalar fval_perturbed = objective(perturbed_x, gradient);
            gradient[i] = (fval - fval_perturbed) / epsilon;
        }
        fval = objective(x, gradient);
    }
};
template<typename Scalar>
class ArmijoLearningRate {
public:
    Scalar operator()(const Function<Scalar>& objective, const Eigen::VectorXd& x, const Eigen::VectorXd& gradient, Scalar fval, size_t iter) const {
       // Armijo rule parameters
        const Scalar beta = 0.8; 
        const Scalar sigma = 0.4; 
        Scalar alpha = 1.0;

        while (true) {
            Eigen::VectorXd perturbed_x = x - alpha * gradient;
            Scalar perturbed_fval = objective(perturbed_x, const_cast<Eigen::VectorXd&>(gradient));
            if (perturbed_fval <= fval - sigma  * gradient.squaredNorm()) {
                break;
            }
            alpha *= beta;
        }

        return alpha;
    }
};

template<typename Scalar>
class ExponentialDecay {
public:
    Scalar operator()(const Function<Scalar>& objective, const Eigen::VectorXd& x, const Eigen::VectorXd& gradient, Scalar fval, size_t iter) const {
       // Exp decay params
        const Scalar mu = 0.8;  
        Scalar alpha = 1.0;


        return alpha*exp(-mu*iter);
    }
};

template<typename Scalar>
class InverseDecay {
public:
    Scalar operator()(const Function<Scalar>& objective, const Eigen::VectorXd& x, const Eigen::VectorXd& gradient, Scalar fval, size_t iter) const {
       // Exp decay params
        const Scalar mu = 0.8;  
        Scalar alpha = 1.0;


        return alpha/(1+mu*iter);
    }
};




template<typename Scalar, typename GradientMethod = CentralDifferences<Scalar>,typename LearningRateMethod = ArmijoLearningRate<Scalar>>
class GradientDescent {
public:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

private:
    Scalar learningRate_;
    size_t maxIterations_;
    Scalar eps_s_;
    Scalar eps_r_;
    GradientMethod gradientMethod_;
    LearningRateMethod learningRateMethod_;



public:
    GradientDescent(Scalar learningRate = 0.01, size_t maxIterations = 1000, Scalar eps_s = 1e-6, Scalar eps_r = 1e-6)
        : learningRate_(learningRate), maxIterations_(maxIterations), eps_s_(eps_s),eps_r_(eps_r) {}

    Vector minimize(Function<Scalar>& objective,
        const Vector& initialGuess) const {
        Vector x = initialGuess;
        Vector prevX = initialGuess;
        Scalar prevFval = objective(initialGuess, prevX);

        for (size_t iter = 0; iter < maxIterations_; ++iter) {
            Scalar fval;
            Vector grad;
            gradientMethod_(objective, x, fval, grad);
            Scalar stepSize = learningRateMethod_(objective, x, grad, fval, iter);
            


            x -= stepSize * grad;
            fval = objective(x, grad);

            // Check for convergence
            if ((prevX - x).norm() < eps_s_) {
                std::cout << "Converged after " << iter << " iterations." << std::endl;
                break;
            }
            if (std::abs(fval-prevFval) < eps_r_) {
                std::cout << "Converged after " << iter << " iterations." << std::endl;
                break;
            }
            if (iter>=maxIterations_){
            std::cout << "Didn't converge :(" << std::endl;
        }
            
            prevX = x;
            prevFval = fval;
        }
        


        return x;
    }
};