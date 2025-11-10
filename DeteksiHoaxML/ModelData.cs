using Microsoft.ML.Data;

// Ganti nama kelas agar sesuai dengan file baru
public class ModelDataInput
{
    // Kolom 0 adalah "Teks"
    // Tanda '?' WAJIB ada karena ada data kosong
    [LoadColumn(0)]
    public string? Teks { get; set; }

    // Kolom 1 adalah "Label"
    // Tanda '?' WAJIB ada karena ada data kosong
    [LoadColumn(1)]
    public string? Label { get; set; }
}

public class ModelDataOutput
{
    // Kolom ini akan berisi prediksi (misal: "SALAH")
    // Nama "PrediksiLabel" akan kita tentukan di pipeline
    public string PrediksiLabel { get; set; } = string.Empty;

    public float[] Score { get; set; } = Array.Empty<float>();
}