// Import library yang kita butuhkan
using Microsoft.ML;

// Definisikan path file
string DATA_FILE_PATH = "dataset_latih.csv";
string MODEL_FILE_PATH = "model.zip"; // Nama file model yang akan kita simpan

// 1. Inisialisasi MLContext
var mlContext = new MLContext(seed: 0);

// 2. Muat Data
Console.WriteLine("Memuat data dari file 'dataset_latih.csv'...");
IDataView dataView = mlContext.Data.LoadFromTextFile<ModelDataInput>(
    path: DATA_FILE_PATH,
    hasHeader: true,
    separatorChar: ';',  // <--- WAJIB KARENA DELIMITER ANDA
    allowQuoting: true
);

// 3. Bersihkan data (WAJIB karena ada data kosong)
// Ini akan berhasil karena ModelDataInput pakai string?
//var cleanDataView = mlContext.Data.FilterRowsByMissingValues(dataView, "Teks", "Label");
Console.WriteLine("Pembersihan data kosong selesai.");

// 4. Bagi Data (80% Latih, 20% Uji)
var trainTestSplit = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
IDataView trainingData = trainTestSplit.TrainSet;
IDataView testData = trainTestSplit.TestSet;

// 5. Definisikan Pipeline Pelatihan (MULTICLASS)
Console.WriteLine("Membangun pipeline model Multiclass...");
var pipeline = mlContext.Transforms.Conversion.MapValueToKey(
        inputColumnName: "Label",   // <-- Kolom "Label" (teks)
        outputColumnName: "Label"   // <-- Ditimpa menjadi "Label" (angka)
    )
    .Append(mlContext.Transforms.Text.FeaturizeText(
        inputColumnName: "Teks",     // <-- Kolom "Teks" (teks)
        outputColumnName: "Features" // <-- Diubah menjadi "Features" (vektor)
    ))
    .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(
        labelColumnName: "Label",
        featureColumnName: "Features"
    ))
    // Konversi kembali prediksi (angka) ke label aslinya (teks)
    .Append(mlContext.Transforms.Conversion.MapKeyToValue(
        inputColumnName: "PredictedLabel", // <-- Output default trainer (angka)
        outputColumnName: "PrediksiLabel"  // <-- Nama properti di ModelDataOutput
    ));

// 6. Latih Model
Console.WriteLine("Memulai pelatihan model...");
var model = pipeline.Fit(trainingData);
Console.WriteLine("Pelatihan model selesai.");

// 7. Evaluasi Model
Console.WriteLine("Mengevaluasi model menggunakan data uji...");
var predictions = model.Transform(testData);
// Gunakan evaluator MULTICLASS
var metrics = mlContext.MulticlassClassification.Evaluate(predictions, "Label");

// 8. Tampilkan Confusion Matrix DULU
Console.WriteLine("--- Confusion Matrix ---");
Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
Console.WriteLine("\n");

// 9. Simpan Model
mlContext.Model.Save(model, trainingData.Schema, MODEL_FILE_PATH);
Console.WriteLine($"Model telah disimpan ke file: {MODEL_FILE_PATH}");

// 10. Uji Coba Prediksi
TestPrediction(mlContext, model);


// 11. TAMPILKAN AKURASI DI PALING AKHIR (YANG PALING PENTING)
Console.WriteLine("\n--- Hasil Evaluasi Model ---");
Console.WriteLine($"* Akurasi (MicroAccuracy): {metrics.MicroAccuracy:P2}");
Console.WriteLine($"* Akurasi (MacroAccuracy): {metrics.MacroAccuracy:P2}");
Console.WriteLine($"* Log-Loss: {metrics.LogLoss:F2}");
Console.WriteLine("--------------------------------------------\n");


// Fungsi untuk Uji Coba Prediksi
static void TestPrediction(MLContext mlContext, ITransformer model)
{
    var predictionEngine = mlContext.Model.CreatePredictionEngine<ModelDataInput, ModelDataOutput>(model);
    
    // Buat data contoh baru
    var sampleNews = new ModelDataInput
    {
        Teks = "Sertifikat Elektronik Itu Rencana Mafia Tanah Hapus Sertifikat SHM"
    };

    var result = predictionEngine.Predict(sampleNews);

    Console.WriteLine("--- Hasil Uji Coba Prediksi ---");
    Console.WriteLine($"Teks: '{sampleNews.Teks}'");
    Console.WriteLine($"Prediksi: {result.PrediksiLabel}");
    Console.WriteLine("---------------------------------");
}