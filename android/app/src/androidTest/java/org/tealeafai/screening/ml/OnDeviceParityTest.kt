package org.tealeafai.screening.ml

import android.graphics.Bitmap
import android.graphics.Color
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import kotlin.math.abs

/**
 * Verifies that the deployed model behaves on the phone the way it did on the desktop.
 *
 * A model can convert cleanly, load without error, and still produce different answers on
 * device: a mismatched label order, a channel swap, or a preprocessing difference all
 * change predictions while throwing nothing. These checks pin the contract that the
 * reported accuracy depends on.
 */
@RunWith(AndroidJUnit4::class)
class OnDeviceParityTest {

    private val appContext get() = InstrumentationRegistry.getInstrumentation().targetContext

    @Test
    fun modelLoadsAndConfigMatchesTheExportRecord() {
        TeaLeafClassifier.create(appContext).use { c ->
            assertEquals("class count must match the trained head", 7, c.config.classOrder.size)
            assertEquals(7, c.config.displayLabelsEn.size)
            assertEquals(7, c.config.displayLabelsId.size)
            assertTrue("temperature must be positive", c.config.temperature > 0f)
            assertTrue(
                "abstention threshold must lie inside the probability range",
                c.config.abstentionThreshold > 0f && c.config.abstentionThreshold < 1f
            )
            // The loader compares the asset's digest against the value recorded at export.
            // Reaching here means the bytes in the APK are the model the metrics belong to.
            assertEquals(64, c.config.modelSha256.length)
        }
    }

    @Test
    fun labelOrderMatchesTheTrainingOrderExactly() {
        // A permuted label list is the classic silent deployment defect: every probability
        // is correct and every name attached to it is wrong.
        val expected = listOf(
            "tea_algal_leaf_spot", "brown_blight", "gray_blight", "helopeltis_damage",
            "red_spider_mite_damage", "green_mirid_bug_damage", "healthy_leaf",
        )
        TeaLeafClassifier.create(appContext).use { c ->
            assertEquals(expected, c.config.classOrder)
        }
    }

    @Test
    fun outputIsAProperDistributionForArbitraryInput() {
        TeaLeafClassifier.create(appContext).use { c ->
            val bmp = Bitmap.createBitmap(224, 224, Bitmap.Config.ARGB_8888)
                .apply { eraseColor(Color.rgb(80, 140, 60)) }
            val p = c.classify(bmp)

            var sum = 0f
            for (v in p.probabilitiesRaw) {
                assertTrue("probability outside [0,1]: $v", v in 0f..1f)
                sum += v
            }
            assertEquals("raw probabilities must sum to 1", 1f, sum, 1e-3f)

            var calSum = 0f
            for (v in p.probabilitiesCalibrated) calSum += v
            assertEquals("calibrated probabilities must sum to 1", 1f, calSum, 1e-3f)

            assertEquals(
                "calibration must not change the ranking",
                p.probabilitiesRaw.indices.maxByOrNull { p.probabilitiesRaw[it] },
                p.topIndex
            )
        }
    }

    @Test
    fun identicalInputProducesIdenticalOutput() {
        // Non-determinism between runs would make every measured latency and accuracy
        // figure unreproducible, and would break the desktop comparison outright.
        TeaLeafClassifier.create(appContext).use { c ->
            val bmp = Bitmap.createBitmap(224, 224, Bitmap.Config.ARGB_8888)
                .apply { eraseColor(Color.rgb(120, 90, 40)) }
            val a = c.classify(bmp).probabilitiesRaw
            val b = c.classify(bmp).probabilitiesRaw
            for (i in a.indices) {
                assertEquals("run-to-run drift at class $i", a[i], b[i], 1e-6f)
            }
        }
    }

    @Test
    fun abstentionFiresExactlyWhenConfidenceIsBelowTheThreshold() {
        TeaLeafClassifier.create(appContext).use { c ->
            val bmp = Bitmap.createBitmap(224, 224, Bitmap.Config.ARGB_8888)
                .apply { eraseColor(Color.rgb(200, 30, 30)) }
            val p = c.classify(bmp)
            assertEquals(
                "abstention must follow the calibrated confidence, not the raw one",
                p.topConfidence < c.config.abstentionThreshold,
                p.abstained
            )
        }
    }

    @Test
    fun activationMapIsPresentAndCorrectlyShaped() {
        TeaLeafClassifier.create(appContext).use { c ->
            val bmp = Bitmap.createBitmap(224, 224, Bitmap.Config.ARGB_8888)
                .apply { eraseColor(Color.rgb(70, 130, 50)) }
            c.classify(bmp)
            val cam = c.lastActivationMap()
            assertTrue("the deployed model should carry the CAM head", cam != null)
            assertEquals("expected a 7x7 grid over 7 classes", 7 * 7 * 7, cam!!.size)
            for (v in cam) assertTrue("non-finite activation: $v", v.isFinite())
        }
    }

    @Test
    fun preprocessingIsIndependentOfSourceResolution() {
        // The eval pipeline resizes the original straight to 224 with no crop, so the same
        // scene photographed at different resolutions must land on the same answer.
        TeaLeafClassifier.create(appContext).use { c ->
            fun solid(w: Int, h: Int): Bitmap =
                Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
                    .apply { eraseColor(Color.rgb(95, 150, 70)) }

            val small = c.classify(solid(224, 224)).probabilitiesRaw
            val large = c.classify(solid(1200, 1200)).probabilitiesRaw
            val maxDev = small.indices.maxOf { abs(small[it] - large[it]) }
            assertTrue(
                "resolution changed the prediction by $maxDev; the resize path differs",
                maxDev < 0.02f
            )
        }
    }

    @Test
    fun writesAParityRecordForTheDesktopComparison() {
        // Emits the on-device probabilities for a set of deterministic synthetic inputs.
        // The same inputs are scored by the desktop runtime, and the two vectors are
        // compared offline; without a shared fixed input there is nothing to compare.
        TeaLeafClassifier.create(appContext).use { c ->
            val report = JSONObject()
            report.put("model_sha256", c.config.modelSha256)
            report.put("variant", c.config.variant)
            report.put("temperature", c.config.temperature.toDouble())
            report.put("abstention_threshold", c.config.abstentionThreshold.toDouble())

            val cases = org.json.JSONArray()
            // Deterministic gradient patterns: reproducible from this description alone.
            for (seedIdx in 0 until 8) {
                val bmp = Bitmap.createBitmap(224, 224, Bitmap.Config.ARGB_8888)
                var seed = (1000L + seedIdx * 7919L)
                for (y in 0 until 224) for (x in 0 until 224) {
                    seed = (seed * 1103515245 + 12345) and 0x7FFFFFFF
                    bmp.setPixel(
                        x, y,
                        Color.rgb((seed % 256).toInt(), ((seed / 256) % 256).toInt(), (x % 256))
                    )
                }
                val p = c.classify(bmp)
                cases.put(JSONObject().apply {
                    put("case", seedIdx)
                    put("probabilities_raw", org.json.JSONArray(p.probabilitiesRaw.map { it.toDouble() }))
                    put("top1_index", p.topIndex)
                    put("top1_confidence_calibrated", p.topConfidence.toDouble())
                    put("abstained", p.abstained)
                    put("inference_ms", p.inferenceMs)
                })
            }
            report.put("cases", cases)

            val out = File(appContext.getExternalFilesDir(null), "parity_ondevice.json")
            out.writeText(report.toString(2))
            println("PARITY_RECORD=${out.absolutePath}")
            assertTrue(out.length() > 100)
        }
    }
}
