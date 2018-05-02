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
    std::cout << "XOR neural network\n";

    //declaration of a neuron of dimension 2
    FormalNeuron<double, 2> hidden_neuron_0;
    //it has 2 weights and 1 bias
    hidden_neuron_0.setBias(-1.5);
    hidden_neuron_0.setWeights({1., 1.});

    //second neuron
    FormalNeuron<double, 2> hidden_neuron_1;
    hidden_neuron_1.setBias(-0.5);
    hidden_neuron_1.setWeights({1., 1.});

    //third neuron
    FormalNeuron<double, 2> output_neuron;
    output_neuron.setBias(-0.5);
    output_neuron.setWeights({-2., 1.});

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
    int nbRandomInputs = 1000;

    for (int i = 0 ; i < nbRandomInputs; ++i)
    {
        //generate random input
        std::array<double, 2> randInput {generate(0., 1.), generate(0., 1.)};
        
        //compute hidden neurons output
        const auto h_0 = hidden_neuron_0.computeOutput(randInput);
        const auto h_1 = hidden_neuron_1.computeOutput(randInput);
        
        //compute output neurons
        //the output of the hidden layer is considered as the input of the output layer
        //this is the place where the relationship between the 3 neurons is defined
        const auto out = output_neuron.computeOutput({h_0, h_1});

        //write input and output in the csv file
        csv_file << randInput[0] << ',' << randInput[1] << ',' << out << '\n';
    }

    csv_file.close();
}
