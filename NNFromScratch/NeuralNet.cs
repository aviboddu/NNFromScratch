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

    public void ApplyDelta(Delta d)
    {
        for (int i = 0; i < neuralNet.Length; i++)
        {
            for (int j = 0; j < neuralNet[i].biases.Length; j++)
                neuralNet[i].biases[j] += d.delta_bias[i][j];
            for (int j = 0; j < neuralNet[i].weights.GetLength(0); j++)
                for (int k = 0; k < neuralNet[i].weights.GetLength(1); k++)
                    neuralNet[i].weights[j, k] += d.delta_weights[i][j, k];
        }
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
            activations[i + 1] = MathUtils.SoftMax(weightedInputs[i]);
        }
        return (activations, weightedInputs);
    }

    public float CalculateClassificationPercentage(LabelImagePair[] data)
    {
        float cost = 0;
        for (int i = 0; i < data.Length; i++)
            cost += CalculateClassification(CalculateOutput(data[i].img), data[i].label);
        return cost / data.Length;
    }

    private static float CalculateClassification(float[] output, float[] label)
    {
        int labelArgMax = 0;
        int outputArgMax = 0;
        float outputMax = float.MinValue;
        for (int i = 0; i < output.Length; i++)
        {
            if (output[i] > outputMax)
            {
                outputMax = output[i];
                outputArgMax = i;
            }

            if (label[i] == 1f)
                labelArgMax = i;
        }
        if (outputArgMax == labelArgMax) return 1;
        return 0;
    }

    public float CalculateTotalCost(LabelImagePair[] data)
    {
        float cost = 0;
        for (int i = 0; i < data.Length; i++)
            cost += CalculateCost(CalculateOutput(data[i].img), data[i].label);
        return cost / data.Length;
    }

    private static float CalculateCost(float[] output, float[] label)
    {
        float cost = 0;
        for (int i = 0; i < output.Length; i++)
            cost -= MathF.Log(output[i] + 1e-5f) * label[i];
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
        (float[][] activations, float[][] weightedInputs) = CalculateActivations(pair.img);
        float[][] errors = new float[neuralNet.Length][];
        errors[^1] = OutputLayerError(pair, weightedInputs, activations);
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
        float[] SigDiv = MathUtils.SoftMaxDerivative(weightedInputs[layer]);
        return MathUtils.HadmardProduct(invErr, SigDiv);
    }

    private static float[] OutputLayerError(LabelImagePair pair, in float[][] weightedInputs,
            in float[][] activations)
    {
        float[] CostGrad = activations[^1].Zip(pair.label, (a, b) => a - b).ToArray();
        float[] SigDiv = MathUtils.SoftMaxDerivative(weightedInputs[^1]);
        return MathUtils.HadmardProduct(CostGrad, SigDiv);
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

    public int TotalWeights() => weights.Length + biases.Length;

    [SetsRequiredMembers]
    public Layer(int layerSize, int inputSize)
    {
        weights = new float[layerSize, inputSize];
        for (int i = 0; i < weights.GetLength(0); i++)
            for (int j = 0; j < weights.GetLength(1); j++)
                weights[i, j] = Random.Shared.NextSingle() * 2f - 1f;
        biases = new float[layerSize];
    }

    public float[] CalculateLayer(float[] input)
    {
        Debug.Assert(input.Length == InputSize);
        return MathUtils.SoftMax(WeightedInput(input));
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
    public float[][] delta_bias = delta_bias;
    public float[][,] delta_weights = delta_weights;

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