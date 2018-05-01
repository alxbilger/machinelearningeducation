#include <iostream>
#include <array>
#include <functional>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <vector>

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
    T computeLinearMapping(const std::array<T,D>& input) const
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
    T computeOutput(const std::array<T,D>& input) const
    {
        T s = computeLinearMapping(input);
        return m_activation(s);
    }

    //setter for the bias
    void setBias(T bias) { m_bias = bias; }

    //setter for the weight
    void setWeights(const std::array<T,D> weights){ m_weights = weights; }

    const std::array<T, D>& getWeights() const { return m_weights;}
    T getBias() const { return m_bias;}
};

template<int D>
struct TrainingData
{
    std::array<double, D> input;
    double classification;
};

template<class T, int D>
void trainNeuron(
    FormalNeuron<T, D>& neuron
    , const std::vector<TrainingData<D> >& trainingData
    , int nbMaxIterations = 20
    , T errorThreshold = 0.)
{
    for(int i = 0; i < nbMaxIterations; ++i)
    {
        for (const auto& t : trainingData)
        {
            const auto y = neuron.computeOutput(t.input);
            auto weights = neuron.getWeights();

            auto inputCoord = std::begin(t.input);
            for (auto& w : weights)
            {
                w += (t.classification - y) * (*inputCoord++);
            }
            auto bias = neuron.getBias();
            bias += (t.classification - y);

            neuron.setWeights(weights);
            neuron.setBias(bias);
        }

        T error = 0.;
        for (const auto& t : trainingData)
            error += std::fabs(t.classification - neuron.computeOutput(t.input));
        error /= static_cast<T>(trainingData.size());

        std::cout << "Iteration error #" << i << ": " << error << '\n';

        if (error < errorThreshold)
        {
            break;
        }
    }
}

int main()
{
    std::cout << "Single formal neuron training\n";

    //random generator seed
    std::srand(std::time(nullptr));

    //tool function to generate a double between a and b
    const auto generate = [](double a, double b)
    {
        return std::min(a, b) + std::abs(a - b) * std::rand() / RAND_MAX;
    }; 

    double a = generate(-10., 10.);
    double b = generate(-10., 10.);

    std::cout << "Random line equation: y = " << a << " * x + " << b << '\n';

    const auto expectedValue = [&a, &b](const std::array<double, 2>& input)
    {
        return (a*input[0] + b - input[1] <= 0.) ? 1. : 0.;
    };

    //number of random input to generate
    int nbRandomInputs = 1000;

    //open a csv file where we will write some data
    std::ofstream csv_file;
    csv_file.open("input_generated.csv");

    std::vector<TrainingData<2> > trainingData;

    //generation of training data by classification
    for (int i = 0 ; i < nbRandomInputs; ++i)
    {
        //generate random input
        std::array<double, 2> randInput {generate(-10., 10.), generate(-10., 10.)};

        //compute classification
        const auto expected = expectedValue(randInput);
        csv_file << randInput[0] << ',' << randInput[1] << ',' << expected << '\n';

        trainingData.push_back({randInput, expected});
    }
    csv_file.close();


    //declaration of a neuron of dimension 2
    FormalNeuron<double, 2> neuron;

    //neuron training
    trainNeuron(neuron, trainingData, 100, 1e-5);

    //print weight and bias after training
    std::cout << "Weights: ";
    for (const auto& w : neuron.getWeights()) 
        std::cout << w << ' ';
    std::cout << '\n';

    std::cout << "Bias: " << neuron.getBias() << '\n';

    //make sure the neuron is able to classify correctly all training data
    for (const auto& t : trainingData)
    {
        const auto y = neuron.computeOutput(t.input);
        if (y != t.classification)
            std::cout << t.input[0] << "," << t.input[1] << '\n';
    }
}
