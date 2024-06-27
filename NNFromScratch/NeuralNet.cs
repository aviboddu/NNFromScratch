using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using MathUtils;

namespace NeuralNet;

public class NeuralNet
{
    private readonly Layer[] neuralNet;
    public NeuralNet(int[] layerSizes)
    {
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
}

public abstract class Layer
{
    public abstract float[] CalculateLayer(float[] input);
}

// // 0.0007 ms to CalculateOutput
public class MatrixLayer : Layer
{
    public Vector<float>[] weights;
    public float[] biases;

    public MatrixLayer(int layerSize, int inputSize)
    {
        weights = new Vector<float>[layerSize];
        for (int i = 0; i < weights.Length; i++)
            weights[i] = new(Random.Shared.NextSingles(inputSize));

        biases = Random.Shared.NextSingles(layerSize);
    }

    public override float[] CalculateLayer(float[] input)
    {
        float[] output = new float[weights.Length];
        Vector<float> inputVec = new(input);
        for (int i = 0; i < weights.Length; i++)
            output[i] = MathUtils.MathUtils.Sigmoid(Vector.Dot(weights[i], inputVec) + biases[i]);
        return output;
    }
}

// 0.0011 ms to CalculateOutput
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
    readonly int num_weights = num_weights;
    public required Vector<float> weights = new(Random.Shared.NextSingles(num_weights));
    public readonly float bias = Random.Shared.NextSingle();

    public float CalculateOutput(float[] input)
    {
        Debug.Assert(input.Length == num_weights, "input.Length != weights.Length");
        return MathUtils.MathUtils.Sigmoid(Vector.Dot(weights, new(input)) + bias);
    }
}