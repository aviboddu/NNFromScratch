using System.Diagnostics;
using System.IO.Compression;

class NNFromScratch
{
    // Client for downloading data
    private static readonly Lazy<HttpClient> client = new(() => new HttpClient()
    {
        BaseAddress = new Uri("https://github.com/HIPS/hypergrad/raw/master/data/mnist/")
    });

    private const string FILE_PREFIX = "./data/"; // File Path
    private const int IMAGE_WIDTH = 28; // Image is square, so width * width dimensions
    private const int IMAGE_SIZE = IMAGE_WIDTH * IMAGE_WIDTH;

    private static readonly string[] file_names = ["train-images-idx3-ubyte", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"];
    private static readonly (float[] label, float[] img)[] training_data = new (float[] label, float[] img)[60000];
    private static readonly (float[] label, float[] img)[] test_data = new (float[] label, float[] img)[10000];

    static int Main(string[] args)
    {
        Stopwatch sw = new(); // TODO: Figure out how to make these measurements more easily
        sw.Start();
        DownloadFiles(false).Wait();
        if (client.IsValueCreated) client.Value.Dispose();
        sw.Stop();
        Debug.WriteLine($"Downloaded files in {sw.ElapsedMilliseconds} ms");

        sw.Restart();
        ExtractFiles(false).Wait();
        sw.Stop();
        Debug.WriteLine($"Extracted files in {sw.ElapsedMilliseconds} ms");

        sw.Restart();
        ParseFiles();
        sw.Stop();
        Debug.WriteLine($"Parsed files in {sw.ElapsedMilliseconds} ms");


        int[] layerSizes = [IMAGE_SIZE, 10, 10, 10];
        sw.Restart();
        NeuralNet.NeuralNet nn = new(layerSizes);
        sw.Stop();
        Debug.WriteLine($"Created Neural Network in {sw.ElapsedMilliseconds} ms");

        int total_iter = 60000;
        sw.Restart();
        for (int i = 0; i < total_iter; i++)
            nn.CalculateOutput(training_data[i].img);
        sw.Stop();
        Debug.WriteLine($"Average time to calculate output of NN is {((float)sw.ElapsedMilliseconds) / total_iter} ms");

        return 0;
    }

    static void ParseFiles()
    {
        // Construct lookup table
        float[] byte_to_float = new float[byte.MaxValue + 1];
        float invByteMax = 1f / byte.MaxValue;
        for (int b = 0; b <= byte.MaxValue; b++)
            byte_to_float[b] = b * invByteMax;

        Parallel.For(0, 2, i =>
        {
            string image_file = string.Concat(FILE_PREFIX, file_names[2 * i]);
            string label_file = string.Concat(FILE_PREFIX, file_names[2 * i + 1]);
            ParseFile(image_file, label_file, i == 0 ? training_data : test_data, byte_to_float);
        });
    }

    static int ReadInt32Flipped(BinaryReader reader)
    {
        byte[] data = reader.ReadBytes(4);
        if (BitConverter.IsLittleEndian) Array.Reverse(data);
        return BitConverter.ToInt32(data);
    }

    static void ParseFile(string image_file, string label_file, (float[] lbl, float[] img)[] image_data, float[] byte_to_float)
    {
        using BinaryReader img_fs = new(File.OpenRead(image_file)), lbl_fs = new(File.OpenRead(label_file));

        // Validating files
        int mgc_num = ReadInt32Flipped(img_fs);
        if (mgc_num != 2051)
            throw new InvalidDataException($"Image file's magic number should be 2051, was {mgc_num}");
        mgc_num = ReadInt32Flipped(lbl_fs);
        if (mgc_num != 2049)
            throw new InvalidDataException($"Label file's magic number should be 2049, was {mgc_num}");
        int num_entries = ReadInt32Flipped(img_fs);
        if (ReadInt32Flipped(lbl_fs) != num_entries)
            throw new InvalidDataException("Label file's number of entries is inequal to image file's number of entries");
        if (ReadInt32Flipped(img_fs) != IMAGE_WIDTH || ReadInt32Flipped(img_fs) != IMAGE_WIDTH)
            throw new InvalidDataException($"Image file's image dimensions are not equal to {IMAGE_WIDTH}");

        // Parsing Files
        byte[] lbls = lbl_fs.ReadBytes(num_entries);
        byte[] img_flat = img_fs.ReadBytes(IMAGE_SIZE * num_entries);
        for (int i = 0; i < num_entries; i++)
        {
            float[] labelVec = LabelToVec(lbls[i]);
            float[] imgVals = ImgBytesToFloat(img_flat[(i * IMAGE_SIZE)..((i + 1) * IMAGE_SIZE)], byte_to_float);
            image_data[i] = new(labelVec, imgVals);
        }
        Debug.WriteLine($"Parsed {image_file} and {label_file}");
    }

    private static float[] ImgBytesToFloat(byte[] img, float[] byte_to_float)
    {
        float[] output = new float[img.Length];
        for (int i = 0; i < img.Length; i++)
            output[i] = byte_to_float[img[i]];
        return output;
    }

    private static float[] LabelToVec(byte label)
    {
        float[] vec = new float[10];
        vec[label] = 1;
        return vec;
    }

    static async Task ExtractFiles(bool overwrite)
    {
        await Parallel.ForEachAsync(file_names, async (file_name, ct) =>
        {
            string compressed_file = string.Concat(FILE_PREFIX, file_name, ".gz");
            string output_file = string.Concat(FILE_PREFIX, file_name);
            if (!overwrite && File.Exists(output_file))
                return;
            if (!File.Exists(compressed_file))
                throw new FileNotFoundException("Files must be downloaded before extraction.", compressed_file);

            using FileStream compressed_fs = File.OpenRead(compressed_file), output_fs = File.Create(output_file);
            using GZipStream gzstream = new(compressed_fs, CompressionMode.Decompress);
            await gzstream.CopyToAsync(output_fs, ct);
            Debug.WriteLine($"Decompressed {output_file}");
        });
    }

    static async Task DownloadFiles(bool overwrite)
    {
        await Parallel.ForEachAsync(file_names, async (file_name, ct) =>
        {
            string uri = string.Concat(file_name, ".gz");
            string file = string.Concat(FILE_PREFIX, file_name, ".gz");
            if (!overwrite && File.Exists(file))
                return;
            _ = Directory.CreateDirectory(Path.GetDirectoryName(file)!);
            File.Create(file).Close();
            Debug.WriteLine($"Downloading File: {uri}");
            await DownloadFile(uri, file, ct);
            Debug.WriteLine($"Downloaded File: {file}");
        });
    }

    static async Task DownloadFile(string uri, string output, CancellationToken ct)
    {
        byte[] file_content = await client.Value.GetByteArrayAsync(uri, ct);
        await File.WriteAllBytesAsync(output, file_content, ct);
    }
}