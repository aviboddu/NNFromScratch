using System.Diagnostics;
using System.IO.Compression;

class NNFromScratch
{
    private static readonly Lazy<HttpClient> client = new(() => new HttpClient()
    {
        BaseAddress = new Uri("https://github.com/HIPS/hypergrad/raw/master/data/mnist/")
    });

    private const string FILE_PREFIX = "./data/";
    private const int IMAGE_WIDTH = 28;
    private static readonly string[] file_names = ["train-images-idx3-ubyte", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"];
    private static readonly Tuple<byte, byte[]>[] training_data = new Tuple<byte, byte[]>[60000];
    private static readonly Tuple<byte, byte[]>[] test_data = new Tuple<byte, byte[]>[10000];

    static int Main(string[] args)
    {
        Stopwatch sw = new(); // Figure out how to make these measurements more easily
        sw.Start();
        DownloadFiles(false).Wait();
        sw.Stop();
        Debug.WriteLine($"Downloaded files in {sw.ElapsedMilliseconds} ms");
        if (client.IsValueCreated) client.Value.Dispose();
        sw.Restart();
        ExtractFiles(false).Wait();
        sw.Stop();
        Debug.WriteLine($"Extracted files in {sw.ElapsedMilliseconds} ms");
        sw.Restart();
        ParseFiles();
        sw.Stop();
        Debug.WriteLine($"Parsed files in {sw.ElapsedMilliseconds} ms");
        return 0;
    }

    static void ParseFiles()
    {
        Parallel.For(0, 2, i =>
        {
            string image_file = string.Concat(FILE_PREFIX, file_names[2 * i]);
            string label_file = string.Concat(FILE_PREFIX, file_names[2 * i + 1]);
            ParseFile(image_file, label_file, i == 0 ? training_data : test_data);
        });
    }

    static int ReadInt32Flipped(BinaryReader reader)
    {
        byte[] data = reader.ReadBytes(4);
        if (BitConverter.IsLittleEndian) Array.Reverse(data);
        return BitConverter.ToInt32(data);
    }

    static void ParseFile(string image_file, string label_file, Tuple<byte, byte[]>[] image_data)
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
        byte[] img_flat = img_fs.ReadBytes(IMAGE_WIDTH * IMAGE_WIDTH * num_entries);
        for (int i = 0; i < num_entries; i++)
            image_data[i] = new Tuple<byte, byte[]>(lbls[i], img_flat[(i * IMAGE_WIDTH * IMAGE_WIDTH)..((i + 1) * IMAGE_WIDTH * IMAGE_WIDTH)]);

        Debug.WriteLine($"Parsed {image_file} and {label_file}");
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