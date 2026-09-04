package org.tealeafai.screening.bench

import android.content.Context
import android.os.Build
import android.os.PowerManager
import android.os.SystemClock
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Test
import org.tensorflow.lite.Interpreter
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

/**
 * On-device latency benchmark for each exported model variant.
 *
 * Protocol choices that determine whether the numbers mean anything:
 *
 * - Timing uses SystemClock.elapsedRealtimeNanos(), which is monotonic and keeps counting
 *   through deep sleep. System.nanoTime() can be adjusted and System.currentTimeMillis()
 *   moves with wall-clock corrections.
 * - Warm-up runs are discarded. The first invocations pay for delegate initialization,
 *   memory arena allocation and CPU frequency ramp-up, and including them inflates the
 *   mean while leaving the median almost untouched -- which is exactly the kind of
 *   discrepancy that makes a reported figure irreproducible.
 * - The median and p95 are reported alongside the mean, because latency distributions on
 *   a phone are right-skewed by scheduler preemption. A mean alone hides that.
 * - Thermal status is read before and after every variant. A benchmark taken while the
 *   device throttles is not comparable to one taken cool, so the state is recorded rather
 *   than assumed constant.
 * - The same fixed input buffer is reused across all measured iterations, so differences
 *   reflect the model and not bitmap preparation. Preprocessing is timed separately by
 *   the application itself.
 *
 * Models are read from the app's external files directory rather than bundled, so all
 * four variants can be measured without shipping 25 MB of assets in the APK.
 */
class LatencyBenchmarkTest {

    private val context: Context
        get() = InstrumentationRegistry.getInstrumentation().targetContext

    private val warmup = 30
    private val measured = 200
    private val repeats = 3

    /** Deterministic input of exactly [bytes], so the workload is identical every run. */
    private fun fixedInputOfBytes(bytes: Int): ByteBuffer {
        val buf = ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder())
        var seed = 12345L
        while (buf.remaining() >= 4) {
            seed = (seed * 1103515245 + 12345) and 0x7FFFFFFF
            buf.putFloat((seed % 256).toFloat())
        }
        while (buf.hasRemaining()) buf.put(0)
        buf.rewind()
        return buf
    }

    /**
     * Build a nested float array matching [shape].
     *
     * Each rank is constructed with its concrete static type. A generic recursion would
     * produce Array<Any>, which erases to Object[] at runtime, and the runtime then rejects
     * it with "cannot resolve DataType" because it cannot see the element type.
     */
    private fun allocateForShape(shape: IntArray): Any = when (shape.size) {
        0 -> FloatArray(1)
        1 -> FloatArray(shape[0])
        2 -> Array(shape[0]) { FloatArray(shape[1]) }
        3 -> Array(shape[0]) { Array(shape[1]) { FloatArray(shape[2]) } }
        4 -> Array(shape[0]) { Array(shape[1]) { Array(shape[2]) { FloatArray(shape[3]) } } }
        else -> throw IllegalArgumentException(
            "unsupported output rank ${shape.size} (${shape.joinToString()})"
        )
    }

    private fun thermalStatus(): Int {
        val pm = context.getSystemService(Context.POWER_SERVICE) as PowerManager
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) pm.currentThermalStatus else -1
    }

    private fun percentile(sorted: DoubleArray, p: Double): Double {
        if (sorted.isEmpty()) return Double.NaN
        val idx = ((sorted.size - 1) * p).toInt().coerceIn(0, sorted.size - 1)
        return sorted[idx]
    }

    private fun benchmarkOne(file: File, useXnnpack: Boolean, threads: Int): JSONObject {
        val result = JSONObject()
        result.put("model_file", file.name)
        result.put("model_bytes", file.length())
        result.put("num_threads", threads)
        result.put("requested_xnnpack", useXnnpack)
        result.put("thermal_before", thermalStatus())

        // Cold-start cost: constructing the interpreter and allocating tensors.
        val loadStart = SystemClock.elapsedRealtimeNanos()
        val options = Interpreter.Options().apply {
            setNumThreads(threads)
            setUseXNNPACK(useXnnpack)
        }
        val interpreter: Interpreter
        try {
            interpreter = Interpreter(file, options)
            // The converted graph carries an unresolved leading dimension, so the input
            // tensor reports a placeholder size until it is resized. Binding it explicitly
            // to a single 224x224x3 image makes the allocated tensor match the buffer that
            // is fed, and fixes the batch size across every variant so the timings compare.
            interpreter.resizeInput(0, intArrayOf(1, 224, 224, 3))
            interpreter.allocateTensors()
        } catch (t: Throwable) {
            // A delegate that cannot prepare a graph is a deployment fact worth recording,
            // not a reason to abandon the whole benchmark.
            result.put("error", "${t::class.java.simpleName}: ${t.message}")
            result.put("usable", false)
            return result
        }
        result.put("cold_start_load_ms", (SystemClock.elapsedRealtimeNanos() - loadStart) / 1e6)
        result.put("usable", true)

        val inShape = interpreter.getInputTensor(0).shape()
        val outShape = interpreter.getOutputTensor(0).shape()
        result.put("input_shape", JSONArray(inShape.toList()))
        result.put("output_shape", JSONArray(outShape.toList()))
        result.put("output_count", interpreter.outputTensorCount)

        // Size the buffer from the tensor's own byte count rather than from an assumed
        // layout, so a variant with a different input convention cannot silently mismatch.
        val inputBytes = interpreter.getInputTensor(0).numBytes()
        result.put("input_bytes", inputBytes)
        val input = fixedInputOfBytes(inputBytes)

        // Allocate outputs for every head, since the CAM-enabled model has two.
        //
        // Shaped primitive arrays are used rather than ByteBuffers on purpose. Resizing the
        // input does not always propagate to the output tensors' reported byte counts, so a
        // ByteBuffer sized from numBytes() can be too small and the runtime overflows it
        // during invocation -- surfacing as a BufferOverflowException from inside the
        // inference call rather than at allocation. Handing the runtime a correctly shaped
        // array lets it bind the sizes itself.
        val outputs = HashMap<Int, Any>()
        for (i in 0 until interpreter.outputTensorCount) {
            outputs[i] = allocateForShape(interpreter.getOutputTensor(i).shape())
        }

        repeat(warmup) {
            input.rewind()
            interpreter.runForMultipleInputsOutputs(arrayOf(input), outputs)
        }

        val repeatBlocks = JSONArray()
        val allSamples = ArrayList<Double>(measured * repeats)

        for (r in 0 until repeats) {
            val samples = DoubleArray(measured)
            for (i in 0 until measured) {
                input.rewind()
                val t0 = SystemClock.elapsedRealtimeNanos()
                interpreter.runForMultipleInputsOutputs(arrayOf(input), outputs)
                samples[i] = (SystemClock.elapsedRealtimeNanos() - t0) / 1e6
            }
            allSamples.addAll(samples.toList())
            val sorted = samples.clone().also { it.sort() }
            val mean = samples.average()
            val sd = sqrt(samples.sumOf { (it - mean) * (it - mean) } / samples.size)
            repeatBlocks.put(JSONObject().apply {
                put("repeat", r)
                put("n", measured)
                put("median_ms", percentile(sorted, 0.50))
                put("mean_ms", mean)
                put("stdev_ms", sd)
                put("p90_ms", percentile(sorted, 0.90))
                put("p95_ms", percentile(sorted, 0.95))
                put("min_ms", sorted.first())
                put("max_ms", sorted.last())
                put("thermal_after_repeat", thermalStatus())
            })
        }

        val pooled = allSamples.toDoubleArray().also { it.sort() }
        val pooledMean = pooled.average()
        result.put("repeats", repeatBlocks)
        result.put("pooled", JSONObject().apply {
            put("n", pooled.size)
            put("median_ms", percentile(pooled, 0.50))
            put("mean_ms", pooledMean)
            put("stdev_ms", sqrt(pooled.sumOf { (it - pooledMean) * (it - pooledMean) } / pooled.size))
            put("p90_ms", percentile(pooled, 0.90))
            put("p95_ms", percentile(pooled, 0.95))
            put("p99_ms", percentile(pooled, 0.99))
            put("min_ms", pooled.first())
            put("max_ms", pooled.last())
        })
        result.put("thermal_after", thermalStatus())

        val native = try { interpreter.lastNativeInferenceDurationNanoseconds / 1e6 } catch (_: Throwable) { -1.0 }
        result.put("last_native_inference_ms", native)

        interpreter.close()
        return result
    }

    @Test
    fun benchmarkAllVariants() {
        // Models are bundled in the TEST APK's assets and staged into the app's cache.
        //
        // The obvious alternative -- adb push into getExternalFilesDir() -- does not work.
        // Android 11+ mediates Android/data through a FUSE layer enforcing per-app access,
        // so a directory created there by the shell user is invisible to the app process
        // even though `adb shell ls` lists it. A Gradle connectedAndroidTest also
        // reinstalls the APK and wipes that directory first.
        //
        // Bundling costs nothing where it matters: the test APK is separate from the app
        // APK, so these 25 MB never ship to a user, and the benchmark becomes
        // self-contained and reproducible by anyone with the repository.
        val testAssets = InstrumentationRegistry.getInstrumentation().context.assets
        val staging = File(context.cacheDir, "benchmark_models").apply { mkdirs() }

        // Scope to this project's models by name prefix. A bare "*.tflite" filter also picks
        // up models bundled by AndroidX test dependencies (notification_classifier_head.tflite
        // among them), which then appear in the report as unrelated failures.
        val assetNames = testAssets.list("")
            ?.filter { it.startsWith("tealeaf_") && it.endsWith(".tflite") }
            ?.sorted().orEmpty()
        val models = assetNames.map { name ->
            val target = File(staging, name)
            // Always overwrite. A "copy only if missing" guard silently benchmarks a stale
            // model after the asset is re-exported: the cache still holds the previous
            // build, the run succeeds, and the reported file size is the only clue that the
            // numbers describe a model that no longer exists.
            testAssets.open(name).use { input ->
                target.outputStream().use { output -> input.copyTo(output) }
            }
            target
        }

        // Fail loudly when nothing is staged. Without this the benchmark iterates an empty
        // list, writes an empty report and reports success -- a green test that measured
        // nothing, which is worse than a red one because it looks like evidence.
        if (models.isEmpty()) {
            throw AssertionError(
                "no .tflite assets found in the test APK. The benchmark measured nothing. " +
                    "Copy the exported variants into app/src/androidTest/assets/ and rebuild."
            )
        }

        val report = JSONObject()
        report.put("device_model", Build.MODEL)
        report.put("device_product", Build.PRODUCT)
        report.put("android_release", Build.VERSION.RELEASE)
        report.put("api_level", Build.VERSION.SDK_INT)
        report.put("supported_abis", JSONArray(Build.SUPPORTED_ABIS.toList()))
        report.put("hardware", Build.HARDWARE)
        report.put("warmup_iterations", warmup)
        report.put("measured_iterations_per_repeat", measured)
        report.put("repeats", repeats)
        report.put("models_found", models.size)
        report.put("protocol", JSONObject().apply {
            put("clock", "SystemClock.elapsedRealtimeNanos (monotonic)")
            put("input", "fixed deterministic buffer reused across all iterations")
            put("excludes", "bitmap decode, rotation and resize; those are timed by the app separately")
            put("thermal", "PowerManager.getCurrentThermalStatus read before and after each variant")
        })

        val results = JSONArray()
        for (m in models) {
            // XNNPACK first; if it cannot prepare the graph the fallback is recorded rather
            // than hidden, because that limitation is itself a deployment finding.
            var r = benchmarkOne(m, useXnnpack = true, threads = 4)
            if (!r.optBoolean("usable", false)) {
                val fallback = benchmarkOne(m, useXnnpack = false, threads = 4)
                fallback.put("xnnpack_failed_error", r.optString("error", ""))
                fallback.put("delegate_used", "none (XNNPACK failed to prepare)")
                r = fallback
            } else {
                r.put("delegate_used", "xnnpack")
            }
            results.put(r)
        }
        report.put("results", results)

        val out = File(context.getExternalFilesDir(null), "benchmark_report.json")
        out.writeText(report.toString(2))

        // Surface a compact summary in the instrumentation log as well, so a failure to
        // pull the file still leaves the numbers recoverable.
        println("BENCHMARK_REPORT_PATH=${out.absolutePath}")
        for (i in 0 until results.length()) {
            val r = results.getJSONObject(i)
            val p = r.optJSONObject("pooled")
            println(
                "BENCH ${r.getString("model_file")} " +
                    "delegate=${r.optString("delegate_used")} " +
                    "load=${"%.1f".format(r.optDouble("cold_start_load_ms", -1.0))}ms " +
                    "median=${"%.3f".format(p?.optDouble("median_ms") ?: -1.0)}ms " +
                    "p95=${"%.3f".format(p?.optDouble("p95_ms") ?: -1.0)}ms " +
                    "thermal=${r.optInt("thermal_before")}->${r.optInt("thermal_after")}"
            )
        }
    }
}
