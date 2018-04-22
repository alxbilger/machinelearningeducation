#include <iostream>
#include <array>
#include <functional>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>

//Heaviside step function
template<class T>
T heaviside(T x)
{
    return (x < 0.) ? 0. : 1.;
}

//Sigmoid function
template<class T>
T sigmoid(T x, T a)
{
    return 1. / (1. + std::exp(-a * x));
}

template<class T, int D>
class FormalNeuron
{
private:
    //weights
    std::array<T, D> m_weights;
    //bias
    T m_bias;

    //activation function
    std::function<T(T)> m_activation;

public:
    FormalNeuron() : m_weights(), m_bias(), m_activation(heaviside<T>) {}

    //compute the linear mapping from a vector of input to a scalar
    T computeLinearMapping(const std::array<T,D>& input)
    {
        T s = m_bias;
        auto weightsIt = std::begin(m_weights);
        auto inputIt = std::begin(input);
        while(weightsIt != std::end(m_weights) && inputIt != std::end(input))
        {
            s += (*weightsIt * *inputIt);
            weightsIt++;
            inputIt++;
        }
        return s;
    }

    //apply the activation function to the result of the linear mapping from an input vector
    T computeOutput(const std::array<T,D>& input)
    {
        T s = computeLinearMapping(input);
        return m_activation(s);
    }

    //setter for the bias
    void setBias(T bias) { m_bias = bias; }

    //setter for the weight
    void setWeights(const std::array<T,D> weights){ m_weights = weights; }
};

int main()
{
    std::cout << "Single formal neuron\n";

    //declaration of a neuron of dimension 2
    FormalNeuron<double, 2> neuron;
    //it has 2 weights and 1 bias
    neuron.setBias(0.5);
    neuron.setWeights({2., -1.5});

    //first example of an input (still in dimension 2)
    std::array<double, 2> input {-2., 8.};
    std::cout << "Input: {" << input[0] << ", " << input[1] << "}\n";

    //print the result of the linear mapping from the input
    const auto linearMapping = neuron.computeLinearMapping(input);
    std::cout << "Linear mapping: " << linearMapping << '\n';

    //print the final output of the neuron from the input
    const auto neuronOutput = neuron.computeOutput(input);
    std::cout << "Neuron output: " << neuronOutput << '\n';

    //tool function to generate a double between a and b
    const auto generate = [](double a, double b)
    {
        return std::min(a, b) + std::abs(a - b) * std::rand() / RAND_MAX;
    }; 

    //open a csv file where we will write some data
    std::ofstream csv_file;
    csv_file.open("classification.csv");

    //random generator seed
    std::srand(std::time(nullptr));

    //number of random input to generate
    int nbRandomInputs = 100;

    for (int i = 0 ; i < nbRandomInputs; ++i)
    {
        //generate random input
        std::array<double, 2> randInput {generate(-10., 10.), generate(-10., 10.)};
        //compute neuron output
        const auto out = neuron.computeOutput(randInput);
        //write input and output in the csv file
        csv_file << randInput[0] << ',' << randInput[1] << ',' << out << '\n';
    }

    csv_file.close();
}
