#include <iostream>
#include <Eigen/Dense>
#include <string>
#include <exprtk.hpp>
#include <algorithm>
/*This header defines the backbone of the parsing and evaluating step for the next algorithms. It's quite flexible, since
it implementates a way of writing potentially n-variables functions*/
template<typename Scalar>
class Function {
public:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef exprtk::symbol_table<Scalar> symbol_table_t;
    typedef exprtk::expression<Scalar> expression_t;
    typedef exprtk::parser<Scalar> parser_t;

    Function(const std::string& expression_string, std::vector<Scalar>& a) : x(a),var(FindVars(expression_string)) {
        const std::string vars = FindVars(expression_string);

        for (auto i = 0; i < vars.size(); ++i) {
            const std::string& curr{ vars[i] };
            symbol_table.add_variable(curr, x[i]);
        }

        symbol_table.add_constants();
        expression.register_symbol_table(symbol_table);

        parser_t parser;
        parser.compile(expression_string, expression);
    }

    Scalar operator()(const Eigen::VectorXd& xval, Eigen::VectorXd&) const {
        set_x(xval);
        return expression.value();
    }
    Scalar get_size(){
        return var.length();
    }

private:
    symbol_table_t symbol_table;
    expression_t expression;
    std::vector<Scalar>& x;
    std::string var;

    void set_x(const Eigen::VectorXd& v2) const {
        x.resize(v2.size());
        for (int i = 0; i < v2.size(); ++i) {
            x[i] = v2[i];
        }
    }

    std::string FindVars(const std::string& expression_string) const {
        // This function should extract variable names from the expression string
        // For simplicity, let's assume variable names are single letters (a, b, c, ...)
        std::string vars;
        for (char c : expression_string) {
            if (isalpha(c) && vars.find(c) == std::string::npos) {
                vars.push_back(c);
            }
        }
        return vars;
    }
};
