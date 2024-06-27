using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;

namespace NeuralNet;

public class NeuralNet
{
    private readonly Node[][] neuralNet;
    public NeuralNet(int[] layerSizes)
    {
        neuralNet = new Node[layerSizes.Length - 1][];
        for (int i = 1; i < layerSizes.Length; i++)
        {
            neuralNet[i - 1] = new Node[layerSizes[i]];
            for (int j = 0; j < layerSizes[i]; j++)
                neuralNet[i - 1][j] = new(layerSizes[i - 1]);
        }
    }

    public float[] CalculateOutput(float[] input)
    {
        for (int i = 0; i < neuralNet.Length; i++)
            input = CalculateLayer(input, i);
        return input;
    }

    public float[] CalculateLayer(float[] input, int nextLayer)
    {
        float[] output = new float[neuralNet[nextLayer].Length];
        for (int i = 0; i < neuralNet[nextLayer].Length; i++)
            output[i] = neuralNet[nextLayer][i].CalculateOutput(input);
        return output;
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

        private static float[] RandomFloats(int num_floats)
        {
            float[] result = new float[num_floats];
            for (int i = 0; i < num_floats; i++)
                result[i] = Random.Shared.NextSingle();
            return result;
        }

        private static float Sigmoid(float x) => 1 / (1 + MathF.Exp(-x));
    }
}