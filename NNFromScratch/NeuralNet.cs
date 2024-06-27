using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using System.Runtime.InteropServices;

namespace NeuralNet;

public class NeuralNet
{
    private readonly Layer[] neuralNet;
    public NeuralNet(int[] layerSizes)
    {
        neuralNet = new Layer[layerSizes.Length - 1];
        for (int i = 1; i < layerSizes.Length; i++)
        {
            neuralNet[i - 1] = new MatrixLayer(layerSizes[i], layerSizes[i - 1]);
        }
    }

    public float[] CalculateOutput(float[] input)
    {
        for (int i = 0; i < neuralNet.Length; i++)
            input = neuralNet[i].CalculateLayer(input);
        return input;
    }

    private static float[] RandomFloats(int num_floats)
    {
        float[] result = new float[num_floats];
        for (int i = 0; i < num_floats; i++)
            result[i] = Random.Shared.NextSingle();
        return result;
    }

    private abstract class Layer
    {
        public abstract float[] CalculateLayer(float[] input);
    }

    private class MatrixLayer : Layer
    {
        public float[,] matrix;
        public float[] biases;

        public MatrixLayer(int layerSize, int inputSize)
        {
            matrix = new float[layerSize, inputSize];
            for (int i = 0; i < matrix.GetLength(0); i++)
                for (int j = 0; j < matrix.GetLength(1); j++)
                    matrix[i, j] = Random.Shared.NextSingle();

            biases = RandomFloats(layerSize);
        }

        public override float[] CalculateLayer(float[] input)
        {
            float[] output = new float[matrix.GetLength(0)];
            Vector<float> inputVec = new(input);
            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                Vector<float> matVec = new(MemoryMarshal.CreateReadOnlySpan(ref matrix[i, 0], matrix.GetLength(1)));
                output[i] = Vector.Dot(matVec, inputVec) + biases[i];
            }
            return output;
        }
    }

    private class NodeLayer : Layer
    {
        public required Node[] nodes;

        [SetsRequiredMembers]
        public NodeLayer(int layerSize, int inputSize)
        {
            nodes = new Node[layerSize];
            for (int i = 0; i < layerSize; i++)
            {
                nodes[i] = new Node(inputSize);
            }
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
    private class Node(int num_weights)
    {
        public required float[] weights = RandomFloats(num_weights);
        public readonly float bias = Random.Shared.NextSingle();

        public float CalculateOutput(float[] input)
        {
            Debug.Assert(input.Length == weights.Length, "input.Length != weights.Length");
            float output = 0;
            for (int i = 0; i < input.Length; i++)
                output += weights[i] * input[i];
            output += bias;
            return Sigmoid(output);
        }

        private static float Sigmoid(float x) => 1 / (1 + MathF.Exp(-x));
    }
}