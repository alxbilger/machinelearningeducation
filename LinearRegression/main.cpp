#include <iostream>
#include <vector>
#include <map>
#include <numeric>

#include <Eigen/Dense>

using HouseProperties = std::map<std::string, double>;

struct House
{
    double price;

    HouseProperties properties;
};

struct Model
{
    double constant;
    std::map<std::string, double> coefs;

    Model() : constant(0.), coefs() {}
};

int main()
{
    std::cout << "Linear regression example\n";

    //selling price of houses
    //from: http://people.sc.fsu.edu/~jburkardt/datasets/regression/x26.txt
    std::vector<House> houses_data_set {
        { 25.9,{ { "bathroom",1.0 },{ "area", 3.4720 },{ "livingArea", 0.998 },{ "garage", 1.0 },{ "room",  7. },{ "bedroom", 4. },{ "age", 42. } } },
        { 29.5,{ { "bathroom",1.0 },{ "area", 3.5310 },{ "livingArea", 1.500 },{ "garage", 2.0 },{ "room",  7. },{ "bedroom", 4. },{ "age", 62. } } },
        { 27.9,{ { "bathroom",1.0 },{ "area", 2.2750 },{ "livingArea", 1.175 },{ "garage", 1.0 },{ "room",  6. },{ "bedroom", 3. },{ "age", 40. } } },
        { 25.9,{ { "bathroom",1.0 },{ "area", 4.0500 },{ "livingArea", 1.232 },{ "garage", 1.0 },{ "room",  6. },{ "bedroom", 3. },{ "age", 54. } } },
        { 29.9,{ { "bathroom",1.0 },{ "area", 4.4550 },{ "livingArea", 1.121 },{ "garage", 1.0 },{ "room",  6. },{ "bedroom", 3. },{ "age", 42. } } },
        { 29.9,{ { "bathroom",1.0 },{ "area", 4.4550 },{ "livingArea", 0.988 },{ "garage", 1.0 },{ "room",  6. },{ "bedroom", 3. },{ "age", 56. } } },
        { 30.9,{ { "bathroom",1.0 },{ "area", 5.8500 },{ "livingArea", 1.240 },{ "garage", 1.0 },{ "room",  7. },{ "bedroom", 3. },{ "age", 51. } } },
        { 28.9,{ { "bathroom",1.0 },{ "area", 9.5200 },{ "livingArea", 1.501 },{ "garage", 0.0 },{ "room",  6. },{ "bedroom", 3. },{ "age", 32. } } },
        { 84.9,{ { "bathroom",2.5 },{ "area", 9.8000 },{ "livingArea", 3.420 },{ "garage", 2.0 },{ "room", 10. },{ "bedroom", 5. },{ "age", 42. } } },
        { 82.9,{ { "bathroom",2.5 },{ "area",12.8000 },{ "livingArea", 3.000 },{ "garage", 2.0 },{ "room",  9. },{ "bedroom", 5. },{ "age", 14. } } },
        { 35.9,{ { "bathroom",1.0 },{ "area", 6.4350 },{ "livingArea", 1.225 },{ "garage", 2.0 },{ "room",  6. },{ "bedroom", 3. },{ "age", 32. } } },
        { 31.5,{ { "bathroom",1.0 },{ "area", 4.9883 },{ "livingArea", 1.552 },{ "garage", 1.0 },{ "room",  6. },{ "bedroom", 3. },{ "age", 30. } } },
        { 31.0,{ { "bathroom",1.0 },{ "area", 5.5200 },{ "livingArea", 0.975 },{ "garage", 1.0 },{ "room",  5. },{ "bedroom", 2. },{ "age", 30. } } },
        { 30.9,{ { "bathroom",1.0 },{ "area", 6.6660 },{ "livingArea", 1.121 },{ "garage", 2.0 },{ "room",  6. },{ "bedroom", 3. },{ "age", 32. } } },
        { 30.0,{ { "bathroom",1.0 },{ "area", 5.0000 },{ "livingArea", 1.020 },{ "garage", 0.0 },{ "room",  5. },{ "bedroom", 2. },{ "age", 46. } } },
        { 28.9,{ { "bathroom",1.0 },{ "area", 9.5200 },{ "livingArea", 1.501 },{ "garage", 0.0 },{ "room",  6. },{ "bedroom", 3. },{ "age", 32. } } },
        { 36.9,{ { "bathroom",1.5 },{ "area", 5.1500 },{ "livingArea", 1.664 },{ "garage", 2.0 },{ "room",  8. },{ "bedroom", 4. },{ "age", 50. } } },
        { 41.9,{ { "bathroom",1.5 },{ "area", 6.9020 },{ "livingArea", 1.488 },{ "garage", 1.5 },{ "room",  7. },{ "bedroom", 3. },{ "age", 22. } } },
        { 40.5,{ { "bathroom",1.5 },{ "area", 7.1020 },{ "livingArea", 1.376 },{ "garage", 1.0 },{ "room",  6. },{ "bedroom", 3. },{ "age", 17. } } },
        { 43.9,{ { "bathroom",1.0 },{ "area", 7.8000 },{ "livingArea", 1.500 },{ "garage", 1.5 },{ "room",  7. },{ "bedroom", 3. },{ "age", 23. } } },
        { 37.5,{ { "bathroom",1.0 },{ "area", 5.5200 },{ "livingArea", 1.256 },{ "garage", 2.0 },{ "room",  6. },{ "bedroom", 3. },{ "age", 40. } } },
        { 37.9,{ { "bathroom",1.5 },{ "area", 4.0000 },{ "livingArea", 1.690 },{ "garage", 1.0 },{ "room",  6. },{ "bedroom", 3. },{ "age", 22. } } },
        { 44.5,{ { "bathroom",1.5 },{ "area", 9.8900 },{ "livingArea", 1.820 },{ "garage", 2.0 },{ "room",  8. },{ "bedroom", 4. },{ "age", 50. } } },
        { 37.9,{ { "bathroom",1.5 },{ "area", 6.7265 },{ "livingArea", 1.652 },{ "garage", 1.0 },{ "room",  6. },{ "bedroom", 3. },{ "age", 44. } } },
        { 38.9,{ { "bathroom",1.5 },{ "area", 9.1500 },{ "livingArea", 1.777 },{ "garage", 2.0 },{ "room",  8. },{ "bedroom", 4. },{ "age", 48. } } },
        { 36.9,{ { "bathroom",1.0 },{ "area", 8.0000 },{ "livingArea", 1.504 },{ "garage", 2.0 },{ "room",  7. },{ "bedroom", 3. },{ "age",  3. } } },
        { 45.8,{ { "bathroom",1.5 },{ "area", 7.3262 },{ "livingArea", 1.831 },{ "garage", 1.5 },{ "room",  8. },{ "bedroom", 4. },{ "age", 31. } } }
        //this observation is not added to the list
        //it will be used to test the linear model
        //{ 41.0,{ { "bathroom",1.5 },{ "area", 5.0000 },{ "livingArea", 1.200 },{ "garage", 2.0 },{ "room",  6. },{ "bedroom", 3. },{ "age", 30. } } }
    };

    std::cout << houses_data_set.size() << " observations\n";

    //function to evaluate the price of a house knowing its properties and using a linear model
    const auto evaluatePrice = [&](const Model& model, const HouseProperties& house)
    {
        double price = model.constant;

        for (const auto& c : model.coefs)
        {
            auto hIt = house.find(c.first);
            if (hIt != house.end())
            {
                price += c.second * hIt->second;
            }
        }
        return price;
    };

    //function to evaluate the error of a model based on the training data
    const auto evaluateError = [&houses_data_set, &evaluatePrice](const Model& model)
    {
        double error = 0.;
        for (const auto& d : houses_data_set)
        {
            const auto p = evaluatePrice(model, d.properties);
            error += std::pow(d.price - p, 2);
        }
        return error / static_cast<double>(houses_data_set.size());
    };

    //consider a model
    Model model; //the model is empty
    model.constant = 3.; //a constant is set
    model.coefs["room"] = 1.; //a coefficient is set for the "room" property
    std::cout << "Error: " << evaluateError(model) << '\n'; //the error of the model is printed

    //definition of the predictors
    std::vector<std::string> variables{ "bathroom", "area", "livingArea", "garage", "room", "bedroom", "age" };

    //definition of the design matrix
    Eigen::MatrixXd X(houses_data_set.size(), variables.size() + 1);
    //definition of the observation vector
    Eigen::VectorXd Y(houses_data_set.size());

    int i = 0;
    for (const auto& d : houses_data_set)
    {
        int j = 0;
        //the first line of the matrix is set to 1 to compute the constant term of the linear model
        X(i, j++) = 1.;

        for (const auto& var : variables)
        {
            auto pIt = d.properties.find(var);
            if (pIt != d.properties.end())
                X(i, j) = pIt->second;
            j++;
        }

        Y(i) = d.price;
        ++i;
    }
    
    //solving the normal equation
    const Eigen::VectorXd B = (X.transpose() * X).ldlt().solve(X.transpose() * Y);

    i = 0;

    //convert the computation result into the model data structure
    model.constant = B(i++);
    std::cout << "Model constant: " << model.constant << '\n';
    model.coefs.clear();
    for (const auto& var : variables)
    {
        model.coefs[var] = B(i++);
        std::cout << "Model (" << var << "): " << model.coefs[var] << '\n';
    }

    //compute the mean price of the observation data
    double meanPrice = std::accumulate(std::begin(houses_data_set), std::end(houses_data_set), 0.,
        [](double a, const House& h) {return a + h.price; }) / static_cast<double>(houses_data_set.size());
    
    std::cout << "Mean price: " << meanPrice << '\n';

    //compute the residual error of the model considering the observation data
    double ssres = evaluateError(model);
    std::cout << "Residual: " << ssres << '\n';

    //compute the total sum of squares (compared to the mean price)
    double sstot = std::accumulate(std::begin(houses_data_set), std::end(houses_data_set), 0.,
        [&meanPrice](double a, const House& h) {return a + std::pow(h.price - meanPrice, 2); }) / static_cast<double>(houses_data_set.size());
    std::cout << "Total sum of squares: " << sstot << '\n';

    //compute the coefficient of determination
    double rSquared = 1. - ssres / sstot;
    std::cout << "Coefficient of determination R^2: " << rSquared << '\n';

    //let's consider another observation, not used to train our model
    const House anotherObservation = { 41., { { "bathroom",1.5 },{ "area", 5.0000 },{ "livingArea", 1.200 },{ "garage", 2.0 },{ "room",  6. },{ "bedroom", 3. },{ "age", 30. } } };

    //estimate its price and compare to the real price
    double estimatedPrice = evaluatePrice(model, anotherObservation.properties);
    std::cout << "Estimate price of a house: " << estimatedPrice << ", compared to " << anotherObservation.price << '\n';

    return 0;
}
