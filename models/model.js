import * as tf from '@tensorflow/tfjs';

let model;

// Memuat model TensorFlow.js dari path yang benar
async function loadModel() {
    model = await tf.loadGraphModel('tfjs_model/model.json');
    console.log('Model berhasil dimuat!');
}

// Fungsi untuk mengonversi gambar menjadi tensor yang sesuai dengan model
function imageToTensor(image) {
    // Mengubah gambar menjadi tensor dengan bentuk [1, height, width, channels]
    return tf.browser.fromPixels(image).resizeNearestNeighbor([177, 177]).toFloat().expandDims(0);
}

// Fungsi untuk mengklasifikasikan gambar
async function classifyImage(image) {
    if (!model) {
        alert("Model belum dimuat!");
        return;
    }

    // Mengonversi gambar menjadi tensor
    const inputTensor = imageToTensor(image);

    // Melakukan inferensi dengan model
    const output = model.predict(inputTensor);

    // Mengambil hasil inferensi
    const outputArray = await output.array();

    // Mengolah dan menampilkan hasil klasifikasi
    displayResults(outputArray[0]);
}

// Fungsi untuk menampilkan hasil klasifikasi
function displayResults(results) {
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '';  // Kosongkan hasil sebelumnya

    // Menampilkan hasil prediksi
    results.forEach((result, index) => {
        resultsDiv.innerHTML += `
            <div class="result">
                <span>Class ${index + 1}: ${result.label.toUpperCase()}</span>
                <div class="bar-container">
                    <div class="bar" style="width: ${result.probability * 100}%"></div>
                </div>
                <span>${(result.probability * 100).toFixed(2)}%</span>
            </div>
        `;
    });
}

// Fungsi untuk menangani pengunggahan gambar dan klasifikasi
function uploadAndClassifyImage() {
    const file = document.getElementById('imageUpload').files[0];
    const reader = new FileReader();

    // Tampilkan gambar yang diunggah
    reader.onloadend = function () {
        const image = document.getElementById('uploadedImage');
        image.src = reader.result;

        const imgElement = document.createElement('img');
        imgElement.src = reader.result;
        imgElement.onload = function () {
            classifyImage(imgElement);
        };
    };

    if (file) {
        reader.readAsDataURL(file);
    }
}

// Memuat model saat halaman dimuat
window.onload = function () {
    loadModel();
};

