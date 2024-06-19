using System.Numerics;

namespace NeuralNet;

public class NeuralNet
{
    private readonly List<float>[][] neuralNet;
    public NeuralNet(int[] layerSizes)
    {
        neuralNet = new List<float>[layerSizes.Length - 1][];
        for (int i = 1; i < layerSizes.Length; i++)
            neuralNet[i - 1] = new List<float>[layerSizes[i]];
    }

    public List<float> CalculateOutput(List<float> input)
    {
        for (int i = 0; i < neuralNet.Length; i++)
        {
            input = CalculateLayer(input, i);
        }
        return input;
    }

    public List<float> CalculateLayer(List<float> input, int nextLayer)
    {
        input.Add(1);
        List<float> output = new(neuralNet[nextLayer].Length);
        for (int i = 0; i < output.Count; i++)
        {
            output[i] = Vector.Dot(new Vector<float>(neuralNet[nextLayer][i].ToArray()), new Vector<float>(input.ToArray()));
        }
        output.Add(1);
        return output;
    }
}