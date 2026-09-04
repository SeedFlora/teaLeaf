package org.tealeafai.screening.ui

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/**
 * Visual language for a field screening instrument.
 *
 * The palette follows the requested dark-tech direction, with one deliberate departure:
 * an amber reserved exclusively for the uncertain state. Reference designs in this style
 * use a single accent for everything, but here "I am not confident enough to answer" must
 * never look like "here is your answer". Rendering both in the same lime would be a
 * design that contradicts the model's own calibration.
 *
 * Numbers are set in monospace throughout. This is an instrument and its confidences,
 * probabilities and latencies are readings, so they get tabular figures that stay aligned
 * as values change rather than a proportional face that makes them shift.
 */
object TeaColors {
    val Base = Color(0xFF0E1310)          // near-black, green-shifted
    val Surface = Color(0xFF161C17)       // raised panel
    val SurfaceHigh = Color(0xFF1D251E)   // panel on panel
    val Outline = Color(0xFF2C3A2D)       // hairline separation

    val Lime = Color(0xFFB8F53D)          // confident answer, primary action
    val LimeDim = Color(0xFF6E9622)       // lime at rest
    val Amber = Color(0xFFFFB020)         // uncertain: never the same colour as an answer
    val AmberDim = Color(0xFF6B4E14)

    val TextHigh = Color(0xFFE8F5E0)
    val TextMid = Color(0xFF9BAE99)
    val TextLow = Color(0xFF6B7A6A)

    val Danger = Color(0xFFFF6B5A)
    val Scrim = Color(0xCC0A0E0B)
}

/** Corner radii: generous on containers, near-circular on controls. */
object TeaShape {
    val card = 28.dp
    val cardSmall = 20.dp
    val pill = 999.dp
    val chip = 14.dp
}

private val Display = TextStyle(
    fontFamily = FontFamily.SansSerif,
    fontWeight = FontWeight.Black,
    fontSize = 34.sp,
    lineHeight = 38.sp,
    letterSpacing = (-1.2).sp,     // tightened: the display face should feel set, not typed
)

private val Mono = FontFamily.Monospace

val TeaTypography = Typography(
    displayLarge = Display,
    headlineMedium = Display.copy(fontSize = 26.sp, lineHeight = 30.sp, letterSpacing = (-0.8).sp),
    titleLarge = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.Bold,
        fontSize = 19.sp,
        letterSpacing = (-0.3).sp,
    ),
    titleMedium = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.SemiBold,
        fontSize = 15.sp,
    ),
    bodyMedium = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.Normal,
        fontSize = 14.sp,
        lineHeight = 20.sp,
    ),
    bodySmall = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.Normal,
        fontSize = 12.sp,
        lineHeight = 17.sp,
    ),
    // Readings: monospace with wide tracking so digits stay legible at small sizes.
    labelLarge = TextStyle(
        fontFamily = Mono,
        fontWeight = FontWeight.Bold,
        fontSize = 13.sp,
        letterSpacing = 1.4.sp,
    ),
    labelSmall = TextStyle(
        fontFamily = Mono,
        fontWeight = FontWeight.Medium,
        fontSize = 11.sp,
        letterSpacing = 1.1.sp,
        textAlign = TextAlign.Start,
    ),
)

@Composable
fun TeaLeafTheme(content: @Composable () -> Unit) {
    // The instrument reads the same in any ambient setting, so the scheme is fixed dark
    // rather than following the system: a light variant would wash out the heatmap overlay
    // that the whole explanation depends on.
    @Suppress("UNUSED_EXPRESSION")
    isSystemInDarkTheme()

    MaterialTheme(
        colorScheme = darkColorScheme(
            primary = TeaColors.Lime,
            onPrimary = TeaColors.Base,
            secondary = TeaColors.Amber,
            background = TeaColors.Base,
            onBackground = TeaColors.TextHigh,
            surface = TeaColors.Surface,
            onSurface = TeaColors.TextHigh,
            surfaceVariant = TeaColors.SurfaceHigh,
            onSurfaceVariant = TeaColors.TextMid,
            outline = TeaColors.Outline,
            error = TeaColors.Danger,
        ),
        typography = TeaTypography,
        content = content,
    )
}
