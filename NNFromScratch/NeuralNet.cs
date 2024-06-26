using System.Diagnostics;

namespace NeuralNet;

public class NeuralNet
{
    private readonly List<float>[][] neuralNet;
    public NeuralNet(int[] layerSizes)
    {
        neuralNet = new List<float>[layerSizes.Length - 1][];
        for (int i = 1; i < layerSizes.Length; i++)
        {
            neuralNet[i - 1] = new List<float>[layerSizes[i]];
            for (int j = 0; j < layerSizes[i]; j++)
            {
                neuralNet[i - 1][j] = RandomList(layerSizes[i - 1] + 1);
            }
        }
    }

    private static List<float> RandomList(int length)
    {
        var list = new List<float>();
        for (int i = 0; i < length; i++)
            list.Add(Random.Shared.NextSingle());
        return list;
    }

    public List<float> CalculateOutput(List<float> input)
    {
        for (int i = 0; i < neuralNet.Length; i++)
        {
            input = CalculateLayer(input, i);
        }
        input.RemoveAt(input.Count - 1);
        return input;
    }

    public List<float> CalculateLayer(List<float> input, int nextLayer)
    {
        input.Add(1);
        List<float> output = new(neuralNet[nextLayer].Length);
        for (int i = 0; i < neuralNet[nextLayer].Length; i++)
            output.Add(Dot(neuralNet[nextLayer][i], input));
        output.Add(1);
        return output;
    }

    private static float Dot(List<float> left, List<float> right)
    {
        float sum = 0;
        foreach ((float x, float y) in left.Zip(right))
        {
            sum += x * y;
        }
        return sum;
    }
}