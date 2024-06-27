using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using Utils;

namespace NeuralNet;

public class NeuralNet
{
    private readonly Layer[] neuralNet;

    public NeuralNet(int[] layerSizes)
    {
        Debug.Assert(layerSizes.All((i) => i > 0), "layerSizes must be greater than 0");
        neuralNet = new Layer[layerSizes.Length - 1];
        for (int i = 1; i < layerSizes.Length; i++)
            neuralNet[i - 1] = new Layer(layerSizes[i], layerSizes[i - 1]);
    }

    public float[] CalculateOutput(float[] input)
    {
        for (int i = 0; i < neuralNet.Length; i++)
            input = neuralNet[i].CalculateLayer(input);
        return input;
    }

    public (float[][] activations, float[][] weightedInputs) CalculateActivations(float[] input)
    {

        float[][] activations = new float[neuralNet.Length + 1][];
        float[][] weightedInputs = new float[neuralNet.Length][];
        activations[0] = input;
        for (int i = 0; i < neuralNet.Length; i++)
        {
            weightedInputs[i] = neuralNet[i].WeightedInput(activations[i]);
            activations[i + 1] = MathUtils.Sigmoid(weightedInputs[i]);
        }
        return (activations, weightedInputs);
    }

    public float CalculateTotalCost(LabelImagePair[] data)
    {
        float cost = 0;
        for (int i = 0; i < data.Length; i++)
        {
            cost += CalculateCost(CalculateOutput(data[i].img), data[i].label);
        }
        return cost / (2 * data.Length);
    }

    private static float CalculateCost(float[] output, float[] label)
    {
        float cost = 0;
        for (int i = 0; i < output.Length; i++)
        {
            float diff = output[i] - label[i];
            cost += diff * diff;
        }
        return cost;
    }

    public Delta CalculateTotalNegativeGradient(LabelImagePair[] data)
    {
        Delta gradient = new(null!, null!);
        foreach (LabelImagePair pair in data)
            gradient.Add(CalculateNegativeGradient(pair));
        gradient.Div(data.Length);
        return gradient;
    }

    public Delta CalculateNegativeGradient(LabelImagePair pair)
    {
        // Output Layer Error: (label-output) * SigmoidDerivative(WeightedInput)
        (float[][] activations, float[][] weightedInputs) = CalculateActivations(pair.img);
        float[][] errors = new float[neuralNet.Length][];
        errors[^1] = OutputLayerError(pair, activations, weightedInputs);
        for (int i = neuralNet.Length - 2; i >= 0; i--)
            errors[i] = LayerError(weightedInputs, i, errors[i + 1]);

        float[][,] delta_w = new float[neuralNet.Length][,];
        for (int i = 0; i < errors.Length; i++)
            delta_w[i] = MathUtils.VecVecToMatrix(errors[i], activations[i]);

        return new(errors, delta_w);
    }

    private float[] LayerError(in float[][] weightedInputs, int layer, float[] nextLayerError)
    {
        float[] invErr = MathUtils.MatMul(MathUtils.MatTranspose(neuralNet[layer + 1].weights), nextLayerError);
        float[] SigDiv = MathUtils.SigmoidDerivative(weightedInputs[layer]);
        return MathUtils.HadmardProduct(invErr, SigDiv).ToArray();
    }

    private float[] OutputLayerError(LabelImagePair pair, in float[][] activations, in float[][] weightedInputs)
    {
        float[] CostGrad = pair.label.Zip(CalculateOutput(pair.img), (a, b) => a - b).ToArray();
        float[] SigDiv = MathUtils.SigmoidDerivative(weightedInputs[^1]);
        return MathUtils.HadmardProduct(CostGrad, SigDiv).ToArray();
    }
}

public struct LabelImagePair
{
    public float[] label;
    public float[] img;
}

// 0.036 ms to CalculateOutput
public class Layer
{
    public int LayerSize => weights.GetLength(0);
    public int InputSize => weights.GetLength(1);

    public required float[,] weights;
    public required float[] biases;

    [SetsRequiredMembers]
    public Layer(int layerSize, int inputSize)
    {
        weights = new float[layerSize, inputSize];
        for (int i = 0; i < layerSize; i++)
            for (int j = 0; j < inputSize; j++)
                weights[i, j] = Random.Shared.NextSingle();

        biases = Random.Shared.NextSingles(layerSize);
    }

    public int TotalWeights() => weights.Length + biases.Length;

    public float[] CalculateLayer(float[] input)
    {
        Debug.Assert(input.Length == InputSize);
        return MathUtils.Sigmoid(WeightedInput(input));
    }

    public float[] WeightedInput(float[] input)
    {
        float[] output = MathUtils.MatMul(weights, input);
        for (int i = 0; i < output.Length; i++)
            output[i] += biases[i];
        return output;
    }
}

public class Delta(float[][] delta_bias, float[][,] delta_weights)
{
    private float[][] delta_bias = delta_bias;
    private float[][,] delta_weights = delta_weights;

    public void Add(Delta other)
    {
        if (delta_bias == null)
            delta_bias = other.delta_bias;
        else
            for (int i = 0; i < delta_bias.Length; i++)
                for (int j = 0; j < delta_bias[i].Length; j++)
                    delta_bias[i][j] += other.delta_bias![i][j];

        if (delta_weights == null)
            delta_weights = other.delta_weights;
        else
            for (int i = 0; i < delta_weights.Length; i++)
                for (int j = 0; j < delta_weights[i].GetLength(0); j++)
                    for (int k = 0; k < delta_weights[i].GetLength(1); k++)
                        delta_weights[i][j, k] += other.delta_weights[i][j, k];

    }

    public void Div(float val)
    {
        float invVal = 1f / val;
        if (delta_bias != null)
            for (int i = 0; i < delta_bias.Length; i++)
                for (int j = 0; j < delta_bias[i].Length; j++)
                    delta_bias[i][j] *= invVal;

        if (delta_weights != null)
            for (int i = 0; i < delta_weights.Length; i++)
                for (int j = 0; j < delta_weights[i].GetLength(0); j++)
                    for (int k = 0; k < delta_weights[i].GetLength(1); k++)
                        delta_weights[i][j, k] *= invVal;
    }
}