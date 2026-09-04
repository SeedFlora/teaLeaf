package org.tealeafai.screening.ml

import android.graphics.Bitmap
import android.graphics.Color
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Rotation checks that require a real Bitmap implementation.
 *
 * Robolectric shadows `Bitmap.createBitmap(src, x, y, w, h, matrix, filter)` by tracking
 * the output dimensions and returning an all-zero pixel buffer, so a JVM test cannot tell
 * a clockwise rotation from an anticlockwise one. Since getting that direction wrong is
 * the exact defect these tests exist to catch -- and it degrades accuracy silently rather
 * than throwing -- the pixel-level assertions run here, on device.
 */
@RunWith(AndroidJUnit4::class)
class PreprocessingInstrumentedTest {

    /** Distinct corner markers, so any rotation is unambiguous. */
    private fun cornerMarked(size: Int = 8): Bitmap {
        val bmp = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        bmp.eraseColor(Color.BLACK)
        bmp.setPixel(0, 0, Color.RED)                 // top-left
        bmp.setPixel(size - 1, 0, Color.GREEN)        // top-right
        bmp.setPixel(size - 1, size - 1, Color.BLUE)  // bottom-right
        bmp.setPixel(0, size - 1, Color.YELLOW)       // bottom-left
        return bmp
    }

    @Test
    fun rotationIsClockwise() {
        // CameraX documents ImageInfo.getRotationDegrees() as the CLOCKWISE rotation that
        // uprights the buffer. Under a clockwise quarter turn the top-left corner travels
        // to the top-right.
        val rotated = Preprocessing.rotateBitmap(cornerMarked(), 90)
        assertEquals("top-left must move to top-right under a clockwise turn",
            Color.RED, rotated.getPixel(rotated.width - 1, 0))
        assertEquals("top-right must move to bottom-right",
            Color.GREEN, rotated.getPixel(rotated.width - 1, rotated.height - 1))
        assertEquals("bottom-left must move to top-left",
            Color.YELLOW, rotated.getPixel(0, 0))
    }

    @Test
    fun rotationIsNotAnticlockwise() {
        // An explicit negative control: the failure mode is a plausible-looking image
        // rotated the wrong way, which no dimension check can detect on a square input.
        val rotated = Preprocessing.rotateBitmap(cornerMarked(), 90)
        assertNotEquals(Color.RED, rotated.getPixel(0, rotated.height - 1))
    }

    @Test
    fun fourQuarterTurnsRestoreEveryCorner() {
        val src = cornerMarked()
        var out = src
        repeat(4) { out = Preprocessing.rotateBitmap(out, 90) }
        assertEquals(Color.RED, out.getPixel(0, 0))
        assertEquals(Color.GREEN, out.getPixel(out.width - 1, 0))
        assertEquals(Color.BLUE, out.getPixel(out.width - 1, out.height - 1))
        assertEquals(Color.YELLOW, out.getPixel(0, out.height - 1))
    }

    @Test
    fun oneEightyMovesOppositeCorners() {
        val out = Preprocessing.rotateBitmap(cornerMarked(), 180)
        assertEquals(Color.RED, out.getPixel(out.width - 1, out.height - 1))
        assertEquals(Color.BLUE, out.getPixel(0, 0))
    }

    @Test
    fun twoSeventyIsTheInverseOfNinety() {
        val src = cornerMarked()
        val out = Preprocessing.rotateBitmap(Preprocessing.rotateBitmap(src, 90), 270)
        assertEquals(Color.RED, out.getPixel(0, 0))
    }

    @Test
    fun resizeScalesRatherThanCrops() {
        // A centre crop would discard the marked corners entirely; a scale preserves them.
        val out = Preprocessing.resizeForModel(cornerMarked(448))
        assertEquals(Preprocessing.INPUT_SIZE, out.width)
        assertEquals(Preprocessing.INPUT_SIZE, out.height)
        assertNotEquals("corner content must survive a resize",
            Color.BLACK, out.getPixel(0, 0))
    }

    @Test
    fun inputBufferPacksRgbInOrderOnDevice() {
        val orange = Color.rgb(200, 120, 40)
        val bmp = Bitmap.createBitmap(224, 224, Bitmap.Config.ARGB_8888).apply { eraseColor(orange) }
        val buf = Preprocessing.toInputBuffer(bmp)
        buf.rewind()
        assertEquals(200f, buf.float, 1e-6f)
        assertEquals(120f, buf.float, 1e-6f)
        assertEquals(40f, buf.float, 1e-6f)
    }
}
