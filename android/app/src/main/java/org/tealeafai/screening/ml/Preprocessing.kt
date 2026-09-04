package org.tealeafai.screening.ml

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import androidx.exifinterface.media.ExifInterface
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Image preparation that must match the desktop training pipeline exactly.
 *
 * Two things here are load-bearing and both are easy to get silently wrong.
 *
 * ROTATION. CameraX's `setTargetRotation` writes metadata only, and `ImageProxy.toBitmap()`
 * does not rotate: it decodes through `BitmapFactory`, which ignores EXIF orientation. A
 * capture therefore arrives rotated with no error anywhere, and the model quietly loses
 * accuracy on sideways leaves. Rotation must be applied explicitly, which is what
 * [rotateBitmap] does, and every entry point routes through it.
 *
 * PIXEL CONVENTION. The exported graph embeds its own normalization, so it consumes raw
 * [0,255] float pixels. Nothing here rescales to [0,1] or subtracts an ImageNet mean --
 * doing so would double-normalize and is precisely the mismatch that produces a model
 * that works on desktop and fails on device.
 *
 * The resize is a plain bilinear scale to the network input size with no crop, mirroring
 * `PIL.Image.resize((224,224), BILINEAR)` used to build the evaluation cache.
 */
object Preprocessing {

    const val INPUT_SIZE = 224
    const val CHANNELS = 3
    const val BYTES_PER_FLOAT = 4

    /** Rotate clockwise by [degrees]; returns the input unchanged when no rotation is needed. */
    fun rotateBitmap(source: Bitmap, degrees: Int): Bitmap {
        val normalized = ((degrees % 360) + 360) % 360
        if (normalized == 0) return source
        val matrix = Matrix().apply { postRotate(normalized.toFloat()) }
        return Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
    }

    /** Map an EXIF orientation tag to the clockwise rotation needed to upright the image. */
    fun exifOrientationToDegrees(orientation: Int): Int = when (orientation) {
        ExifInterface.ORIENTATION_ROTATE_90 -> 90
        ExifInterface.ORIENTATION_ROTATE_180 -> 180
        ExifInterface.ORIENTATION_ROTATE_270 -> 270
        else -> 0
    }

    /**
     * Decode a gallery image and upright it using its EXIF tag.
     *
     * Gallery images differ from camera captures: the file may carry an orientation tag
     * that the decoder ignores. Reading it explicitly keeps both paths equivalent.
     */
    fun decodeAndUpright(imageStream: InputStream, exifStream: InputStream): Bitmap {
        val degrees = exifOrientationToDegrees(
            ExifInterface(exifStream).getAttributeInt(
                ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL
            )
        )
        val decoded = BitmapFactory.decodeStream(imageStream)
            ?: throw IllegalArgumentException("image could not be decoded")
        return rotateBitmap(decoded, degrees)
    }

    /** Bilinear resize to the square network input. No crop, matching the desktop pipeline. */
    fun resizeForModel(source: Bitmap): Bitmap =
        if (source.width == INPUT_SIZE && source.height == INPUT_SIZE) source
        else Bitmap.createScaledBitmap(source, INPUT_SIZE, INPUT_SIZE, /* filter = */ true)

    /**
     * Pack a 224x224 bitmap into the float32 NHWC buffer the interpreter expects.
     *
     * Values stay in [0,255]. Channel order is R,G,B to match the training pipeline, which
     * fed `PIL.Image.convert("RGB")` output. Android packs pixels as ARGB in a single Int,
     * so the shifts below extract R, G, B in that order; swapping them would silently
     * degrade a colour-sensitive task where lesion hue separates classes.
     */
    fun toInputBuffer(bitmap: Bitmap): ByteBuffer {
        require(bitmap.width == INPUT_SIZE && bitmap.height == INPUT_SIZE) {
            "expected ${INPUT_SIZE}x$INPUT_SIZE, got ${bitmap.width}x${bitmap.height}"
        }
        val buffer = ByteBuffer
            .allocateDirect(INPUT_SIZE * INPUT_SIZE * CHANNELS * BYTES_PER_FLOAT)
            .order(ByteOrder.nativeOrder())

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        for (pixel in pixels) {
            buffer.putFloat(((pixel shr 16) and 0xFF).toFloat())   // R
            buffer.putFloat(((pixel shr 8) and 0xFF).toFloat())    // G
            buffer.putFloat((pixel and 0xFF).toFloat())            // B
        }
        buffer.rewind()
        return buffer
    }

    /** Convenience: upright, resize and pack in the order the model requires. */
    fun prepare(source: Bitmap, rotationDegrees: Int): ByteBuffer =
        toInputBuffer(resizeForModel(rotateBitmap(source, rotationDegrees)))
}
