package org.tealeafai.screening.ml

import android.graphics.Bitmap
import android.graphics.Color
import androidx.exifinterface.media.ExifInterface
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.nio.ByteBuffer

/**
 * Preprocessing tests.
 *
 * The rotation cases exist because CameraX hands back an unrotated bitmap plus a rotation
 * value, and forgetting to apply it produces no error at all -- only quietly worse
 * predictions on any photograph not taken in the sensor's native orientation.
 *
 * The buffer tests pin the two conventions the model depends on: values stay in [0,255]
 * because normalization lives inside the exported graph, and channels are packed R,G,B.
 */
@RunWith(RobolectricTestRunner::class)
// Robolectric 4.14.1 ships Android runtimes up to API 35, while the app targets 36. Left
// unpinned it refuses to start with "targetSdkVersion=36 > maxSdkVersion=35". The pin
// affects only the simulated runtime used for these graphics helpers, none of which
// depends on API 36 behaviour.
@Config(sdk = [35])
class PreprocessingTest {

    private fun solid(w: Int, h: Int, color: Int): Bitmap =
        Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888).apply { eraseColor(color) }

    /** An asymmetric image, so a rotation cannot pass unnoticed. */
    private fun cornerMarked(w: Int, h: Int): Bitmap {
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        bmp.eraseColor(Color.BLACK)
        bmp.setPixel(0, 0, Color.RED)
        bmp.setPixel(w - 1, 0, Color.GREEN)
        return bmp
    }

    @Test
    fun `zero rotation returns the same instance`() {
        val src = solid(10, 20, Color.BLUE)
        assertTrue(Preprocessing.rotateBitmap(src, 0) === src)
        assertTrue(Preprocessing.rotateBitmap(src, 360) === src)
    }

    @Test
    fun `ninety degree rotation swaps the dimensions`() {
        val src = solid(10, 20, Color.BLUE)
        val rotated = Preprocessing.rotateBitmap(src, 90)
        assertEquals(20, rotated.width)
        assertEquals(10, rotated.height)
    }

    @Test
    fun `one hundred eighty degree rotation preserves the dimensions`() {
        val src = solid(10, 20, Color.BLUE)
        val rotated = Preprocessing.rotateBitmap(src, 180)
        assertEquals(10, rotated.width)
        assertEquals(20, rotated.height)
    }

    // Pixel-level rotation checks are NOT here, and that is deliberate. Robolectric's
    // shadow of Bitmap.createBitmap(src, x, y, w, h, matrix, filter) tracks only the
    // resulting dimensions; it returns an all-zero pixel buffer without applying the
    // matrix. Asserting pixel positions here fails against the shadow even when the code
    // is correct, so those assertions live in
    // androidTest/.../PreprocessingInstrumentedTest, where Bitmap is the real
    // implementation. Weakening them to pass under Robolectric would delete the only
    // check that catches an anticlockwise rotation.

    @Test
    fun `negative and oversized angles are normalized`() {
        val src = solid(10, 20, Color.BLUE)
        assertEquals(20, Preprocessing.rotateBitmap(src, -90).width)
        assertEquals(20, Preprocessing.rotateBitmap(src, 450).width)
        assertTrue(Preprocessing.rotateBitmap(src, 720) === src)
    }

    @Test
    fun `four successive quarter turns restore the original dimensions`() {
        // Pixel identity after four turns is asserted in the instrumented test; only the
        // dimension round-trip is verifiable against Robolectric's bitmap shadow.
        val src = cornerMarked(8, 16)
        var out: Bitmap = src
        repeat(4) { out = Preprocessing.rotateBitmap(out, 90) }
        assertEquals(src.width, out.width)
        assertEquals(src.height, out.height)
    }

    @Test
    fun `exif orientation maps to the correct rotation`() {
        assertEquals(0, Preprocessing.exifOrientationToDegrees(ExifInterface.ORIENTATION_NORMAL))
        assertEquals(90, Preprocessing.exifOrientationToDegrees(ExifInterface.ORIENTATION_ROTATE_90))
        assertEquals(180, Preprocessing.exifOrientationToDegrees(ExifInterface.ORIENTATION_ROTATE_180))
        assertEquals(270, Preprocessing.exifOrientationToDegrees(ExifInterface.ORIENTATION_ROTATE_270))
        assertEquals(0, Preprocessing.exifOrientationToDegrees(ExifInterface.ORIENTATION_UNDEFINED))
    }

    @Test
    fun `resize produces the exact network input size`() {
        for (dims in listOf(100 to 100, 1200 to 1600, 300 to 200, 224 to 224)) {
            val out = Preprocessing.resizeForModel(solid(dims.first, dims.second, Color.GREEN))
            assertEquals(Preprocessing.INPUT_SIZE, out.width)
            assertEquals(Preprocessing.INPUT_SIZE, out.height)
        }
    }

    @Test
    fun `resize does not crop, it scales the whole frame`() {
        // A centre crop would discard the marked corners; a scale keeps them.
        val src = cornerMarked(448, 448)
        val out = Preprocessing.resizeForModel(src)
        assertNotEquals(Color.BLACK, out.getPixel(0, 0))
    }

    @Test
    fun `input buffer has the exact expected capacity`() {
        val buffer = Preprocessing.toInputBuffer(solid(224, 224, Color.WHITE))
        assertEquals(224 * 224 * 3 * 4, buffer.capacity())
    }

    @Test
    fun `pixel values stay in the zero to two-fifty-five range`() {
        // The exported graph embeds its own normalization, so rescaling here would
        // double-normalize and silently degrade every prediction.
        val buffer: ByteBuffer = Preprocessing.toInputBuffer(solid(224, 224, Color.WHITE))
        buffer.rewind()
        repeat(3) { assertEquals(255f, buffer.float, 1e-6f) }

        val black = Preprocessing.toInputBuffer(solid(224, 224, Color.BLACK))
        black.rewind()
        repeat(3) { assertEquals(0f, black.float, 1e-6f) }
    }

    @Test
    fun `channels are packed in red green blue order`() {
        // Android stores ARGB in one Int. Extracting them in the wrong order would swap
        // red and blue, which matters here because lesion hue separates several classes.
        val orange = Color.rgb(200, 120, 40)
        val buffer = Preprocessing.toInputBuffer(solid(224, 224, orange))
        buffer.rewind()
        assertEquals(200f, buffer.float, 1e-6f)
        assertEquals(120f, buffer.float, 1e-6f)
        assertEquals(40f, buffer.float, 1e-6f)
    }

    @Test(expected = IllegalArgumentException::class)
    fun `packing rejects a bitmap of the wrong size`() {
        Preprocessing.toInputBuffer(solid(100, 100, Color.WHITE))
    }

    @Test
    fun `full prepare path yields a correctly sized buffer regardless of rotation`() {
        val src = solid(1200, 1600, Color.rgb(10, 20, 30))
        for (deg in listOf(0, 90, 180, 270)) {
            assertEquals(224 * 224 * 3 * 4, Preprocessing.prepare(src, deg).capacity())
        }
    }
}
