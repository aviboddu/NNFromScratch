using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.InteropServices;
using MathUtils;

namespace NeuralNet;

public class NeuralNet
{
    private readonly Layer[] neuralNet;

    public NeuralNet(int[] layerSizes)
    {
        Debug.Assert(layerSizes.All((i) => i > 0), "layerSizes must be greater than 0");
        neuralNet = new Layer[layerSizes.Length - 1];
        for (int i = 1; i < layerSizes.Length; i++)
            neuralNet[i - 1] = new MatrixLayer(layerSizes[i], layerSizes[i - 1]);
    }

    public float[] CalculateOutput(float[] input)
    {
        for (int i = 0; i < neuralNet.Length; i++)
            input = neuralNet[i].CalculateLayer(input);
        return input;
    }

    public float CalculateTotalCost(LabelImagePair[] data)
    {
        float cost = 0;
        for (int i = 0; i < data.Length; i++)
        {
            cost += CalculateCost(data[i]);
        }
        return cost / data.Length;
    }

    private float CalculateCost(LabelImagePair pair)
    {
        float[] output = CalculateOutput(pair.img);
        float cost = 0;
        for (int i = 0; i < output.Length; i++)
        {
            float diff = output[i] - pair.label[i];
            cost += diff * diff;
        }
        return cost;
    }

    public float[] CalculateTotalNegativeGradient(LabelImagePair[] data)
    {
        float[] gradient = new float[neuralNet.Sum((l) => l.TotalWeights())];
        foreach (LabelImagePair pair in data)
        {
            float[] component_grad = CalculateNegativeGradient(pair);
            for (int i = 0; i < gradient.Length; i++)
                gradient[i] += component_grad[i];
        }
        for (int i = 0; i < gradient.Length; i++)
            gradient[i] /= data.Length;
        return gradient;
    }

    public float[] CalculateNegativeGradient(LabelImagePair pair)
    {
        return new float[neuralNet.Sum((n) => n.TotalWeights())];
    }

}

public struct LabelImagePair
{
    public float[] label;
    public float[] img;
}

public abstract class Layer
{
    public abstract float[] CalculateLayer(float[] input);
    public abstract int TotalWeights();
}

// 0.036 ms to CalculateOutput
public class MatrixLayer : Layer
{
    private int LayerSize => weights.GetLength(0);
    private int InputSize => weights.GetLength(1);

    public required float[,] weights;
    public required float[] biases;

    [SetsRequiredMembers]
    public MatrixLayer(int layerSize, int inputSize)
    {
        weights = new float[layerSize, inputSize];
        for (int i = 0; i < layerSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                weights[i, j] = Random.Shared.NextSingle();
            }
        }

        biases = Random.Shared.NextSingles(layerSize);
    }

    public override int TotalWeights() => weights.Length + biases.Length;

    public override float[] CalculateLayer(float[] input)
    {
        Debug.Assert(input.Length == InputSize);
        float[] output = new float[LayerSize];
        for (int i = 0; i < output.Length; i++)
        {
            float dotProd = MathUtils.MathUtils.Dot(MemoryMarshal.CreateSpan(ref weights[i, 0], input.Length), input);
            output[i] = MathUtils.MathUtils.Sigmoid(dotProd + biases[i]);
        }
        return output;
    }
}

// 0.036 ms to CalculateOutput
public class NodeLayer : Layer
{
    public required Node[] nodes;

    [SetsRequiredMembers]
    public NodeLayer(int layerSize, int inputSize)
    {
        nodes = new Node[layerSize];
        for (int i = 0; i < layerSize; i++)
            nodes[i] = new Node(inputSize);
    }

    public override int TotalWeights() => nodes.Sum((i) => i.weights.Length) + nodes.Length;

    public override float[] CalculateLayer(float[] input)
    {
        float[] output = new float[nodes.Length];
        for (int i = 0; i < nodes.Length; i++)
            output[i] = nodes[i].CalculateOutput(input);
        return output;
    }
}

[method: SetsRequiredMembers]
public class Node(int num_weights)
{
    public required float[] weights = Random.Shared.NextSingles(num_weights);
    public readonly float bias = Random.Shared.NextSingle();

    public float CalculateOutput(float[] input)
    {
        return MathUtils.MathUtils.Sigmoid(MathUtils.MathUtils.Dot(weights, input) + bias);
    }
}