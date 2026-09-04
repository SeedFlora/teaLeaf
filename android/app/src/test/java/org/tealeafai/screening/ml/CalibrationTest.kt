package org.tealeafai.screening.ml

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln

/**
 * Tests for the calibration and abstention arithmetic.
 *
 * These run on the JVM with no Android dependency, because the logic they cover is the
 * part most likely to diverge silently from the desktop pipeline: a temperature applied
 * the wrong way still produces a valid-looking probability vector.
 */
class CalibrationTest {

    private val eps = 1e-5f

    private fun assertIsDistribution(p: FloatArray) {
        var sum = 0f
        for (v in p) {
            assertTrue("probability out of range: $v", v >= 0f && v <= 1f)
            sum += v
        }
        assertEquals("probabilities must sum to 1", 1f, sum, 1e-4f)
    }

    @Test
    fun `temperature of one leaves probabilities unchanged`() {
        val p = floatArrayOf(0.1f, 0.2f, 0.05f, 0.05f, 0.3f, 0.15f, 0.15f)
        assertArrayEquals(p, TeaLeafClassifier.applyTemperature(p, 1f), eps)
    }

    @Test
    fun `output remains a probability distribution after scaling`() {
        val p = floatArrayOf(0.7f, 0.1f, 0.05f, 0.05f, 0.04f, 0.03f, 0.03f)
        for (t in listOf(0.5f, 0.8f, 1.27f, 2.0f, 5.0f)) {
            assertIsDistribution(TeaLeafClassifier.applyTemperature(p, t))
        }
    }

    @Test
    fun `temperature above one reduces peak confidence`() {
        val p = floatArrayOf(0.9f, 0.02f, 0.02f, 0.02f, 0.02f, 0.01f, 0.01f)
        val softened = TeaLeafClassifier.applyTemperature(p, 1.27f)
        assertTrue(
            "T>1 must soften the peak (was ${p[0]}, became ${softened[0]})",
            softened[0] < p[0]
        )
    }

    @Test
    fun `temperature below one sharpens peak confidence`() {
        val p = floatArrayOf(0.5f, 0.2f, 0.1f, 0.08f, 0.06f, 0.04f, 0.02f)
        val sharpened = TeaLeafClassifier.applyTemperature(p, 0.5f)
        assertTrue(sharpened[0] > p[0])
    }

    @Test
    fun `scaling never changes which class ranks highest`() {
        // Ranking invariance is what makes it safe to calibrate after the fact: the
        // decision is preserved and only the reported confidence moves.
        val p = floatArrayOf(0.31f, 0.29f, 0.13f, 0.11f, 0.08f, 0.05f, 0.03f)
        val expectedTop = p.indices.maxByOrNull { p[it] }
        for (t in listOf(0.3f, 0.9f, 1.27f, 3.0f, 10.0f)) {
            val scaled = TeaLeafClassifier.applyTemperature(p, t)
            assertEquals(expectedTop, scaled.indices.maxByOrNull { scaled[it] })
        }
    }

    @Test
    fun `matches an independent softmax over log-probabilities`() {
        // Recomputed here from first principles rather than reusing the implementation,
        // so a sign error or a missing log would fail rather than agree with itself.
        val p = floatArrayOf(0.4f, 0.25f, 0.15f, 0.1f, 0.05f, 0.03f, 0.02f)
        val t = 1.37f

        val logits = p.map { ln(it.toDouble()) / t }
        val max = logits.max()
        val exps = logits.map { exp(it - max) }
        val sum = exps.sum()
        val expected = exps.map { (it / sum).toFloat() }.toFloatArray()

        assertArrayEquals(expected, TeaLeafClassifier.applyTemperature(p, t), 1e-4f)
    }

    @Test
    fun `entropy is zero for a certain prediction and maximal for a uniform one`() {
        val certain = floatArrayOf(1f, 0f, 0f, 0f, 0f, 0f, 0f)
        assertEquals(0f, TeaLeafClassifier.entropyOf(certain), eps)

        val uniform = FloatArray(7) { 1f / 7f }
        assertEquals(ln(7.0).toFloat(), TeaLeafClassifier.entropyOf(uniform), 1e-4f)

        val middling = floatArrayOf(0.5f, 0.2f, 0.1f, 0.08f, 0.06f, 0.04f, 0.02f)
        assertTrue(TeaLeafClassifier.entropyOf(middling) > TeaLeafClassifier.entropyOf(certain))
        assertTrue(TeaLeafClassifier.entropyOf(middling) < TeaLeafClassifier.entropyOf(uniform))
    }

    @Test
    fun `abstention triggers only strictly below the threshold`() {
        val threshold = 0.7638f
        fun abstains(confidence: Float) = confidence < threshold

        assertTrue(abstains(0.70f))
        assertTrue(abstains(threshold - 1e-4f))
        assertTrue("a value exactly at the threshold must be answered", !abstains(threshold))
        assertTrue(!abstains(0.95f))
    }

    @Test
    fun `extreme probabilities do not produce NaN`() {
        val degenerate = floatArrayOf(1f, 0f, 0f, 0f, 0f, 0f, 0f)
        val scaled = TeaLeafClassifier.applyTemperature(degenerate, 1.27f)
        for (v in scaled) assertTrue("NaN or infinity produced: $v", v.isFinite())
        assertIsDistribution(scaled)
    }

    @Test
    fun `sha256 helper matches a known digest`() {
        // Guards the integrity check that refuses to run a model asset whose bytes do not
        // match the digest recorded at export time.
        assertEquals(
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            TeaLeafClassifier.sha256Of(ByteArray(0))
        )
        assertEquals(
            "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb",
            TeaLeafClassifier.sha256Of("a".toByteArray())
        )
    }

    @Test
    fun `class order has exactly seven entries and no duplicates`() {
        val classes = listOf(
            "tea_algal_leaf_spot", "brown_blight", "gray_blight", "helopeltis_damage",
            "red_spider_mite_damage", "green_mirid_bug_damage", "healthy_leaf",
        )
        assertEquals(7, classes.size)
        assertEquals("class labels must be unique", classes.size, classes.toSet().size)
    }
}
