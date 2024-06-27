namespace MathUtils;

public static class MathUtils
{
    public static float[] NextSingles(this Random rand, int num_floats)
    {
        float[] result = new float[num_floats];
        for (int i = 0; i < num_floats; i++)
            result[i] = rand.NextSingle();
        return result;
    }

    public static float Sigmoid(float x) => 1 / (1 + MathF.Exp(-x));
}