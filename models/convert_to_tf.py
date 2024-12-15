import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

# Memuat model ONNX
def load_onnx_model(path):
    try:
        return onnx.load(path)
    except Exception as e:
        print(f"Gagal memuat model ONNX: {e}")
        return None

# Mengonversi model ONNX ke format TensorFlow
def convert_onnx_to_tf(onnx_model):
    try:
        return prepare(onnx_model)
    except Exception as e:
        print(f"Gagal mengonversi model ONNX: {e}")
        return None

# Menyimpan model TensorFlow dalam format .pb
def save_tf_model(tf_rep, output_path):
    try:
        tf_rep.export_graph(output_path)
    except Exception as e:
        print(f"Gagal menyimpan model TensorFlow: {e}")

# Memuat model TensorFlow yang telah disimpan
def load_tf_model(path):
    try:
        return tf.saved_model.load(path)
    except Exception as e:
        print(f"Gagal memuat model TensorFlow: {e}")
        return None

# Membuat tensor input untuk inferensi
def generate_input_tensor(shape):
    try:
        return tf.random.normal(shape)
    except Exception as e:
        print(f"Gagal membuat input tensor: {e}")
        return None

# Melakukan inferensi menggunakan model TensorFlow
def run_inference(saved_model, input_tensor):
    try:
        infer = saved_model.signatures["serving_default"]
        output = infer(input=tf.convert_to_tensor(input_tensor))
        return output
    except Exception as e:
        print(f"Gagal melakukan inferensi: {e}")
        return None

# Fungsi utama untuk menjalankan proses
if __name__ == "__main__":
    onnx_model_path = "model2.onnx"
    output_path = "model2_tf"

    # Langkah 1: Memuat model ONNX
    onnx_model = load_onnx_model(onnx_model_path)
    if onnx_model is None:
        exit()

    # Langkah 2: Mengonversi model ONNX ke TensorFlow
    tf_rep = convert_onnx_to_tf(onnx_model)
    if tf_rep is None:
        exit()

    # Langkah 3: Menyimpan model TensorFlow
    save_tf_model(tf_rep, output_path)

    # Langkah 4: Memuat kembali model TensorFlow
    saved_model = load_tf_model(output_path)
    if saved_model is None:
        exit()

    print("Model TensorFlow berhasil dimuat.")

    # Menampilkan tanda tangan model
    print("Signatures yang tersedia:")
    print(saved_model.signatures)

    # Langkah 5: Menyediakan input tensor untuk inferensi
    input_shape = [1, 3, 177, 177]  # Gunakan dimensi input yang sesuai
    input_tensor = generate_input_tensor(input_shape)
    if input_tensor is None:
        exit()

    # Langkah 6: Melakukan inferensi
    output = run_inference(saved_model, input_tensor)
    if output is not None:
        print("Hasil inferensi:")
        print(output)

        # Menampilkan kunci output untuk memastikan nama yang benar
        print("Kunci output:", output.keys())

        # Langkah 7: Mengonversi output menjadi probabilitas dan prediksi kelas
        output_key = list(output.keys())[0]  # Ambil kunci pertama
        probabilities = tf.nn.softmax(output[output_key]).numpy()
        print("Probabilitas:", probabilities)

        predicted_class = tf.argmax(probabilities, axis=1).numpy()
        print("Prediksi kelas:", predicted_class)
