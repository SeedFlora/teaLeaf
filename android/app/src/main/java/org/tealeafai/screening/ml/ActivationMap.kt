package org.tealeafai.screening.ml

import android.graphics.Bitmap
import android.graphics.Color
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * Renders a class activation map into a heatmap the user can see over their photograph.
 *
 * The map arrives from the model as a 7x7 grid per class -- the spatial resolution of the
 * final convolutional feature map -- and is upsampled bilinearly to the display size.
 * That coarseness is inherent to the method and is why the overlay indicates a region
 * rather than an outline; presenting it as a crisp segmentation would imply a precision
 * the 7x7 grid does not have.
 *
 * The overlay answers "which part of this image drove the prediction". It does not answer
 * "why", and nothing in this class attempts to generate a biological rationale, because
 * the network did not compute one.
 */
object ActivationMap {

    /** Extract one class's map from the flat [1,H,W,C] tensor the interpreter returns. */
    fun extractClassMap(flat: FloatArray, height: Int, width: Int, numClasses: Int, classIndex: Int): Array<FloatArray> {
        require(classIndex in 0 until numClasses) { "class index out of range: $classIndex" }
        require(flat.size == height * width * numClasses) {
            "expected ${height * width * numClasses} values, got ${flat.size}"
        }
        return Array(height) { y ->
            FloatArray(width) { x -> flat[(y * width + x) * numClasses + classIndex] }
        }
    }

    /**
     * Clamp negatives away and rescale to [0,1].
     *
     * Negative activations argue *against* the class, so they are dropped rather than
     * mapped to a colour that would read as weak evidence for it. Normalization is
     * per-map, which makes the overlay a relative statement about this image only -- two
     * images' heat intensities are not comparable.
     */
    fun normalize(map: Array<FloatArray>): Array<FloatArray> {
        var peak = 0f
        for (row in map) for (v in row) peak = max(peak, v)
        if (peak <= 0f) return Array(map.size) { FloatArray(map[0].size) }
        return Array(map.size) { y -> FloatArray(map[0].size) { x -> max(0f, map[y][x]) / peak } }
    }

    /** Bilinear upsample of the coarse grid to an arbitrary output size. */
    fun upsample(map: Array<FloatArray>, outHeight: Int, outWidth: Int): Array<FloatArray> {
        val inH = map.size
        val inW = map[0].size
        val out = Array(outHeight) { FloatArray(outWidth) }
        // Half-pixel centres, so the grid is not biased toward the top-left corner.
        for (y in 0 until outHeight) {
            val sy = ((y + 0.5f) * inH / outHeight - 0.5f).coerceIn(0f, (inH - 1).toFloat())
            val y0 = sy.toInt().coerceAtMost(inH - 1)
            val y1 = (y0 + 1).coerceAtMost(inH - 1)
            val fy = sy - y0
            for (x in 0 until outWidth) {
                val sx = ((x + 0.5f) * inW / outWidth - 0.5f).coerceIn(0f, (inW - 1).toFloat())
                val x0 = sx.toInt().coerceAtMost(inW - 1)
                val x1 = (x0 + 1).coerceAtMost(inW - 1)
                val fx = sx - x0
                val top = map[y0][x0] * (1 - fx) + map[y0][x1] * fx
                val bottom = map[y1][x0] * (1 - fx) + map[y1][x1] * fx
                out[y][x] = top * (1 - fy) + bottom * fy
            }
        }
        return out
    }

    /**
     * Map an intensity in [0,1] to a perceptually ordered colour.
     *
     * Transparent through yellow to red, with alpha rising alongside intensity so cool
     * regions stay legible underneath instead of being flooded with colour.
     */
    fun heatColor(intensity: Float): Int {
        val t = intensity.coerceIn(0f, 1f)
        if (t < 0.15f) return Color.TRANSPARENT
        val alpha = (200 * ((t - 0.15f) / 0.85f)).roundToInt().coerceIn(0, 200)
        val r: Int
        val g: Int
        if (t < 0.6f) {
            val k = (t - 0.15f) / 0.45f          // yellow ramp
            r = (255 * k).roundToInt().coerceIn(0, 255)
            g = (220 * k).roundToInt().coerceIn(0, 255)
        } else {
            val k = (t - 0.6f) / 0.4f            // yellow to red
            r = 255
            g = (220 * (1 - k)).roundToInt().coerceIn(0, 255)
        }
        return Color.argb(alpha, r, g, 0)
    }

    /** Compose the heatmap over a copy of the photograph. */
    fun overlay(photo: Bitmap, camFlat: FloatArray, camH: Int, camW: Int,
                numClasses: Int, classIndex: Int): Bitmap {
        val normalized = normalize(extractClassMap(camFlat, camH, camW, numClasses, classIndex))
        val scaled = upsample(normalized, photo.height, photo.width)

        val out = photo.copy(Bitmap.Config.ARGB_8888, true)
        val row = IntArray(photo.width)
        for (y in 0 until photo.height) {
            out.getPixels(row, 0, photo.width, 0, y, photo.width, 1)
            for (x in 0 until photo.width) {
                val heat = heatColor(scaled[y][x])
                if (heat != Color.TRANSPARENT) row[x] = blend(row[x], heat)
            }
            out.setPixels(row, 0, photo.width, 0, y, photo.width, 1)
        }
        return out
    }

    private fun blend(base: Int, over: Int): Int {
        val a = Color.alpha(over) / 255f
        return Color.rgb(
            ((1 - a) * Color.red(base) + a * Color.red(over)).roundToInt().coerceIn(0, 255),
            ((1 - a) * Color.green(base) + a * Color.green(over)).roundToInt().coerceIn(0, 255),
            ((1 - a) * Color.blue(base) + a * Color.blue(over)).roundToInt().coerceIn(0, 255),
        )
    }

    /**
     * Fraction of the activation mass falling outside a centred region.
     *
     * Every training image is a detached leaf centred on plain paper, so a model can score
     * well by reading the backdrop. When most of the evidence sits at the frame edges the
     * prediction deserves suspicion, and surfacing that is more useful than a confident
     * label. This is a diagnostic signal, not a correctness measure.
     */
    fun peripheralMass(map: Array<FloatArray>, centerFraction: Float = 0.6f): Float {
        val h = map.size
        val w = map[0].size
        val my = ((h * (1 - centerFraction)) / 2).roundToInt()
        val mx = ((w * (1 - centerFraction)) / 2).roundToInt()
        var total = 0f
        var peripheral = 0f
        for (y in 0 until h) for (x in 0 until w) {
            val v = max(0f, map[y][x])
            total += v
            if (y < my || y >= h - my || x < mx || x >= w - mx) peripheral += v
        }
        return if (total <= 0f) 0f else peripheral / total
    }
}
