package org.tealeafai.screening.data

import android.content.Context
import android.graphics.Bitmap
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import org.json.JSONArray
import org.json.JSONObject
import org.tealeafai.screening.ml.TeaLeafClassifier
import java.io.File
import java.security.MessageDigest
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone
import java.util.UUID

/**
 * Append-only field log, written to app-specific storage for later retrieval over ADB.
 *
 * The record keeps three things structurally apart: what the device MEASURED, what a human
 * JUDGED, and whether a qualified specialist later VERIFIED it. That separation is the
 * reason the log is useful at all -- confidence distribution, abstention rate and latency
 * can be analysed without reading a single label, so those results hold regardless of who
 * annotated. Only label-dependent analyses are restricted to the annotated subset, and
 * accuracy only to the expert-verified one.
 *
 * See android/FIELD_LOG_SCHEMA.md for the full contract.
 */
class FieldLogger(private val context: Context) {

    private val root: File = File(context.getExternalFilesDir(null), "research").apply { mkdirs() }
    private val images: File = File(root, "images").apply { mkdirs() }
    private val log = File(root, "field_log.jsonl")
    private val sessionId = UUID.randomUUID().toString()
    private var captureIndex = 0

    private val iso = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.US).apply {
        timeZone = TimeZone.getTimeZone("UTC")
    }

    val logPath: String get() = log.absolutePath

    private fun sha256(bytes: ByteArray) =
        MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }

    /**
     * Write one capture. Returns the record id so a later verdict can be attached to it.
     *
     * The image is saved alongside and its digest recorded, so a transfer that truncates a
     * file is detectable rather than silently changing what a record refers to.
     */
    fun record(
        photo: Bitmap,
        prediction: TeaLeafClassifier.Prediction,
        config: TeaLeafClassifier.ModelConfig,
        source: String,
        rotationApplied: Int,
        peripheralMass: Float,
        consentGranted: Boolean,
    ): String {
        captureIndex += 1
        val recordId = UUID.randomUUID().toString()
        val stamp = iso.format(Date())

        val imageFile = File(images, "img_${System.currentTimeMillis()}_$captureIndex.jpg")
        val bytes = java.io.ByteArrayOutputStream().use { out ->
            photo.compress(Bitmap.CompressFormat.JPEG, 92, out)
            out.toByteArray()
        }
        imageFile.writeBytes(bytes)

        val pm = context.getSystemService(Context.POWER_SERVICE) as PowerManager
        val bm = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager

        val measurement = JSONObject().apply {
            put("model", JSONObject().apply {
                put("file", config.modelAsset)
                put("sha256", config.modelSha256)
                put("variant", config.variant)
                put("version", config.version)
                put("input_size", 224)
                put("input_convention", "raw [0,255] float32; normalization embedded in the graph")
                put("class_order", JSONArray(config.classOrder))
            })
            put("probabilities_raw", JSONArray(prediction.probabilitiesRaw.map { it.toDouble() }))
            put("probabilities_calibrated", JSONArray(prediction.probabilitiesCalibrated.map { it.toDouble() }))
            put("temperature", config.temperature.toDouble())
            put("top1_index", prediction.topIndex)
            put("top1_confidence_calibrated", prediction.topConfidence.toDouble())
            put("top3_indices", JSONArray(prediction.top3Indices.toList()))
            put("predictive_entropy", prediction.entropy.toDouble())
            put("abstained", prediction.abstained)
            put("abstention_threshold", config.abstentionThreshold.toDouble())
            put("cam_peripheral_mass", peripheralMass.toDouble())
            put("latency_ms", JSONObject().apply {
                put("preprocess", prediction.preprocessMs)
                put("inference", prediction.inferenceMs)
                put("postprocess", prediction.postprocessMs)
                put("end_to_end", prediction.preprocessMs + prediction.inferenceMs + prediction.postprocessMs)
            })
            put("runtime", JSONObject().apply {
                put("delegate", if (config.useXnnpack) "xnnpack" else "cpu")
                put("num_threads", config.numThreads)
            })
            put("device_state", JSONObject().apply {
                put("thermal_status",
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) pm.currentThermalStatus else -1)
                put("battery_percent", bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY))
                put("device_model", Build.MODEL)
                put("android_release", Build.VERSION.RELEASE)
                put("api_level", Build.VERSION.SDK_INT)
            })
        }

        val record = JSONObject().apply {
            put("schema_version", "1.0")
            put("record_id", recordId)
            put("session_id", sessionId)
            put("capture_index", captureIndex)
            put("timestamp_utc", stamp)
            put("image", JSONObject().apply {
                put("file", "images/${imageFile.name}")
                put("sha256", sha256(bytes))
                put("width", photo.width)
                put("height", photo.height)
                put("source", source)
                put("rotation_applied_degrees", rotationApplied)
            })
            put("measurement", measurement)
            // Empty until the user answers; kept as its own object so an analysis can
            // filter on its presence rather than inferring from a sentinel value.
            put("annotation", JSONObject().apply {
                put("verdict", JSONObject.NULL)
                put("claimed_true_class", JSONObject.NULL)
                put("annotator_expertise", "unstated")
                put("annotated_at_utc", JSONObject.NULL)
            })
            put("expert_verification", JSONObject().apply {
                put("verified", false)
                put("expert_role", JSONObject.NULL)
                put("expert_class", JSONObject.NULL)
            })
            put("consent", JSONObject().apply {
                put("research_use_granted", consentGranted)
                put("granted_at_utc", stamp)
            })
        }

        log.appendText(record.toString() + "\n")
        return recordId
    }

    /**
     * Append a verdict as a separate line referring to the original record.
     *
     * Rewriting the original line in place would mean rewriting the whole file on every
     * answer, risking corruption of earlier captures if the write is interrupted in the
     * field. An append-only journal keeps every measurement durable; the analysis script
     * folds verdicts onto records by id.
     */
    fun recordVerdict(recordId: String, verdict: String, claimedTrueClass: String?, expertise: String) {
        val entry = JSONObject().apply {
            put("schema_version", "1.0")
            put("record_type", "annotation_update")
            put("record_id", recordId)
            put("annotation", JSONObject().apply {
                put("verdict", verdict)
                put("claimed_true_class", claimedTrueClass ?: JSONObject.NULL)
                put("annotator_expertise", expertise)
                put("annotated_at_utc", iso.format(Date()))
            })
        }
        log.appendText(entry.toString() + "\n")
    }

    fun recordCount(): Int = if (log.exists()) log.readLines().count { it.isNotBlank() } else 0
}
