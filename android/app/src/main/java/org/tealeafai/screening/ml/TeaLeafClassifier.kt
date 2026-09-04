package org.tealeafai.screening.ml

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.os.SystemClock
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.security.MessageDigest

/**
 * On-device tea-leaf health classifier.
 *
 * Note the import: LiteRT kept the Java package `org.tensorflow.lite` when the Maven
 * coordinate moved to `com.google.ai.edge.litert`. Only the dependency line changed.
 *
 * Delegates: the V2 Interpreter exposes exactly four options -- threads, XNNPACK,
 * cancellable, runtime. `addDelegate`, `NnApiDelegate` and the `Delegate` interface were
 * removed in V2, and NNAPI itself was deprecated in Android 15 and is not declared as a
 * feature by the benchmark device. CPU with XNNPACK is therefore not a fallback here; it
 * is the only path the API offers.
 *
 * Thread count defaults to 4 rather than the core count: the Snapdragon 8 Gen 2 is a
 * heterogeneous 1+4+3 design, and spilling onto efficiency cores costs more in
 * synchronization than it returns.
 */
class TeaLeafClassifier private constructor(
    private val interpreter: Interpreter,
    val config: ModelConfig,
) : AutoCloseable {

    data class ModelConfig(
        val modelAsset: String,
        val modelSha256: String,
        val modelBytes: Long,
        val variant: String,
        val version: String,
        val classOrder: List<String>,
        val displayLabelsEn: List<String>,
        val displayLabelsId: List<String>,
        val temperature: Float,
        val abstentionThreshold: Float,
        val numThreads: Int,
        val useXnnpack: Boolean,
    )

    data class Prediction(
        val probabilitiesRaw: FloatArray,
        val probabilitiesCalibrated: FloatArray,
        val topIndex: Int,
        val topConfidence: Float,
        val top3Indices: IntArray,
        val entropy: Float,
        val abstained: Boolean,
        val preprocessMs: Double,
        val inferenceMs: Double,
        val postprocessMs: Double,
    )

    /** Run the full path on an already-uprighted bitmap. */
    fun classify(source: Bitmap, rotationDegrees: Int = 0): Prediction {
        val t0 = SystemClock.elapsedRealtimeNanos()
        val input = Preprocessing.prepare(source, rotationDegrees)
        val t1 = SystemClock.elapsedRealtimeNanos()

        val raw = runInference(input)
        val t2 = SystemClock.elapsedRealtimeNanos()

        val calibrated = applyTemperature(raw, config.temperature)
        var top = 0
        for (i in calibrated.indices) if (calibrated[i] > calibrated[top]) top = i
        val order = calibrated.indices.sortedByDescending { calibrated[it] }
        val t3 = SystemClock.elapsedRealtimeNanos()

        return Prediction(
            probabilitiesRaw = raw,
            probabilitiesCalibrated = calibrated,
            topIndex = top,
            topConfidence = calibrated[top],
            top3Indices = order.take(3).toIntArray(),
            entropy = entropyOf(calibrated),
            abstained = calibrated[top] < config.abstentionThreshold,
            preprocessMs = (t1 - t0) / 1e6,
            inferenceMs = (t2 - t1) / 1e6,
            postprocessMs = (t3 - t2) / 1e6,
        )
    }

    /** Flattened class activation maps from the most recent inference, or null. */
    private var lastCam: FloatArray? = null

    /**
     * Class activation maps for the last classified image, flattened as [h][w][class].
     *
     * Present only when the loaded model was exported with the CAM head. The maps come
     * from the graph's second output, so they cost one tensor read rather than a second
     * forward pass.
     */
    fun lastActivationMap(): FloatArray? = lastCam

    /** Pure interpreter time, used by the benchmark harness to exclude bitmap work. */
    fun runInference(input: ByteBuffer): FloatArray {
        input.rewind()
        val n = config.classOrder.size

        if (interpreter.outputTensorCount < 2) {
            val output = Array(1) { FloatArray(n) }
            interpreter.run(input, output)
            lastCam = null
            return output[0]
        }

        // Two heads: probabilities and the activation maps. Which index carries which is
        // read from the tensor ranks rather than assumed, because the converter does not
        // guarantee output ordering.
        val outputs = HashMap<Int, Any>()
        var probIndex = 0
        var camIndex = -1
        for (i in 0 until interpreter.outputTensorCount) {
            val shape = interpreter.getOutputTensor(i).shape()
            if (shape.size == 2) {
                probIndex = i
                outputs[i] = Array(shape[0]) { FloatArray(shape[1]) }
            } else if (shape.size == 4) {
                camIndex = i
                outputs[i] = Array(shape[0]) {
                    Array(shape[1]) { Array(shape[2]) { FloatArray(shape[3]) } }
                }
            } else {
                outputs[i] = FloatArray(shape.fold(1) { a, b -> a * b })
            }
        }

        interpreter.runForMultipleInputsOutputs(arrayOf(input), outputs)

        @Suppress("UNCHECKED_CAST")
        val probs = (outputs[probIndex] as Array<FloatArray>)[0]

        lastCam = if (camIndex >= 0) {
            @Suppress("UNCHECKED_CAST")
            val cam = (outputs[camIndex] as Array<Array<Array<FloatArray>>>)[0]
            val h = cam.size
            val w = cam[0].size
            val c = cam[0][0].size
            FloatArray(h * w * c).also { flat ->
                for (y in 0 until h) for (x in 0 until w) for (k in 0 until c) {
                    flat[(y * w + x) * c + k] = cam[y][x][k]
                }
            }
        } else null

        return probs
    }

    /** Nanoseconds the native runtime reported for the last invocation, if available. */
    fun lastNativeInferenceNanos(): Long =
        try {
            interpreter.lastNativeInferenceDurationNanoseconds
        } catch (_: Throwable) {
            -1L
        }

    override fun close() = interpreter.close()

    companion object {
        const val CONFIG_ASSET = "model_config.json"

        /**
         * Temperature scaling applied to probabilities.
         *
         * The exported graph ends in softmax, so logits are recovered as log(p) before
         * dividing by T. Scaling the probabilities directly would be a different and
         * incorrect operation. Softmax is invariant to an additive constant, so the
         * reconstruction loses nothing.
         */
        fun applyTemperature(probs: FloatArray, temperature: Float): FloatArray {
            if (temperature <= 0f || temperature == 1f) return probs.copyOf()
            val scaled = FloatArray(probs.size)
            var max = Float.NEGATIVE_INFINITY
            for (i in probs.indices) {
                scaled[i] = (Math.log(probs[i].coerceAtLeast(1e-12f).toDouble()) / temperature).toFloat()
                if (scaled[i] > max) max = scaled[i]
            }
            var sum = 0f
            for (i in scaled.indices) {
                scaled[i] = Math.exp((scaled[i] - max).toDouble()).toFloat()
                sum += scaled[i]
            }
            for (i in scaled.indices) scaled[i] /= sum
            return scaled
        }

        fun entropyOf(probs: FloatArray): Float {
            var h = 0.0
            for (p in probs) if (p > 0f) h -= p * Math.log(p.toDouble())
            return h.toFloat()
        }

        fun sha256Of(bytes: ByteArray): String =
            MessageDigest.getInstance("SHA-256").digest(bytes)
                .joinToString("") { "%02x".format(it) }

        private fun mapAsset(assets: AssetManager, name: String): MappedByteBuffer {
            val fd = assets.openFd(name)
            FileInputStream(fd.fileDescriptor).use { stream ->
                return stream.channel.map(
                    FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength
                )
            }
        }

        /**
         * Load the model named by the bundled config.
         *
         * The config carries the SHA-256 recorded at export time, and the loaded bytes are
         * hashed and compared. A mismatch means the asset in the APK is not the model the
         * reported accuracy belongs to, which would invalidate every on-device number, so
         * it fails loudly instead of running the wrong weights.
         */
        fun create(context: Context, overrideVariantAsset: String? = null): TeaLeafClassifier {
            val assets = context.assets
            val json = JSONObject(assets.open(CONFIG_ASSET).bufferedReader().use { it.readText() })

            val asset = overrideVariantAsset ?: json.getString("model_asset")
            val expectedSha = json.optString("model_sha256", "")

            val bytes = assets.open(asset).use { it.readBytes() }
            val actualSha = sha256Of(bytes)
            if (expectedSha.isNotEmpty() && !actualSha.equals(expectedSha, ignoreCase = true)) {
                throw IllegalStateException(
                    "model asset $asset does not match the recorded digest; " +
                        "expected $expectedSha but found $actualSha"
                )
            }

            fun labels(key: String): List<String> {
                val arr = json.getJSONArray(key)
                return (0 until arr.length()).map { arr.getString(it) }
            }

            val numThreads = json.optInt("num_threads", 4)
            val useXnnpack = json.optBoolean("use_xnnpack", true)

            val options = Interpreter.Options().apply {
                setNumThreads(numThreads)
                setUseXNNPACK(useXnnpack)
            }

            val config = ModelConfig(
                modelAsset = asset,
                modelSha256 = actualSha,
                modelBytes = bytes.size.toLong(),
                variant = json.optString("variant", "unknown"),
                version = json.optString("version", "unknown"),
                classOrder = labels("class_order"),
                displayLabelsEn = labels("display_labels_en"),
                displayLabelsId = labels("display_labels_id"),
                temperature = json.optDouble("temperature", 1.0).toFloat(),
                abstentionThreshold = json.optDouble("abstention_threshold", 0.0).toFloat(),
                numThreads = numThreads,
                useXnnpack = useXnnpack,
            )

            return TeaLeafClassifier(Interpreter(mapAsset(assets, asset), options), config)
        }
    }
}
