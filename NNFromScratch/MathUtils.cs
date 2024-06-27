using System.Diagnostics;
using System.Runtime.InteropServices;

namespace Utils;

public static class MathUtils
{
    public static float[] NextSingles(this Random rand, int num_floats)
    {
        float[] result = new float[num_floats];
        for (int i = 0; i < num_floats; i++)
            result[i] = rand.NextSingle();
        return result;
    }

    public static float Dot(Span<float> a, Span<float> b)
    {
        Span<float> prod = HadmardProduct(a, b);
        float output = 0;
        foreach (float s in prod)
        {
            output += s;
        }
        return output;
    }

    public static float Sigmoid(float x) => 1 / (1 + MathF.Exp(-x));

    public static float[] Sigmoid(IEnumerable<float> v) => v.Select(Sigmoid).ToArray();

    public static float SigmoidDerivative(float x)
    {
        float ex = MathF.Exp(x);
        return ex / ((ex + 1) * (ex + 1));
    }

    public static float[] SigmoidDerivative(IEnumerable<float> v) => v.Select(SigmoidDerivative).ToArray();

    public static float[,] MatTranspose(float[,] matrix)
    {
        float[,] output = new float[matrix.GetLength(1), matrix.GetLength(0)];
        for (int i = 0; i < matrix.GetLength(0); i++)
            for (int j = 0; j < matrix.GetLength(1); j++)
                output[j, i] = matrix[i, j];
        return output;
    }

    public static float[] MatMul(float[,] matrix, float[] vec)
    {
        Debug.Assert(matrix.GetLength(1) == vec.Length);
        float[] output = new float[matrix.GetLength(0)];
        for (int i = 0; i < output.Length; i++)
        {
            output[i] = Dot(MemoryMarshal.CreateSpan(ref matrix[i, 0], vec.Length), vec);
        }
        return output;
    }

    public static float[,] VecVecToMatrix(float[] x, float[] y)
    {
        float[,] output = new float[x.Length, y.Length];
        for (int i = 0; i < x.Length; i++)
            for (int j = 0; j < y.Length; j++)
                output[i, j] = x[i] * y[j];
        return output;
    }

    public static Span<float> HadmardProduct(Span<float> x, Span<float> y)
    {
        for (int i = 0; i < x.Length; i++)
        {
            x[i] *= y[i];
        }
        return x;
    }
}