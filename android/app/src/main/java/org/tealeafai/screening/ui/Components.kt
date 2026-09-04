package org.tealeafai.screening.ui

import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.foundation.Canvas
import kotlin.math.cos
import kotlin.math.sin

/**
 * The signature element: confidence as an arc with a physical notch at the abstention
 * threshold.
 *
 * A plain progress bar would render 0.62 and 0.82 as two similar-looking fills. Here the
 * threshold is a gap cut into the track, so a reading either clears it or visibly does
 * not, and the colour changes across that boundary. It makes the model's own decision
 * rule legible instead of hiding it behind a percentage, which is the whole point of
 * shipping calibrated abstention rather than a bare argmax.
 */
@Composable
fun ConfidenceRing(
    confidence: Float,
    threshold: Float,
    abstained: Boolean,
    modifier: Modifier = Modifier,
    size: Dp = 168.dp,
) {
    val animated by animateFloatAsState(
        targetValue = confidence.coerceIn(0f, 1f),
        animationSpec = tween(700),
        label = "confidence",
    )
    val accent = if (abstained) TeaColors.Amber else TeaColors.Lime
    val accentDim = if (abstained) TeaColors.AmberDim else TeaColors.LimeDim

    Box(modifier = modifier.size(size), contentAlignment = Alignment.Center) {
        Canvas(Modifier.fillMaxSize()) {
            val stroke = 14.dp.toPx()
            val inset = stroke / 2 + 6.dp.toPx()
            val arcSize = androidx.compose.ui.geometry.Size(
                this.size.width - inset * 2, this.size.height - inset * 2
            )
            val topLeft = androidx.compose.ui.geometry.Offset(inset, inset)

            // The arc spans 260 degrees, leaving a gap at the bottom so the scale reads as
            // a gauge rather than a closed loop.
            val startAngle = 140f
            val sweepTotal = 260f

            drawArc(
                color = TeaColors.Outline,
                startAngle = startAngle,
                sweepAngle = sweepTotal,
                useCenter = false,
                topLeft = topLeft,
                size = arcSize,
                style = Stroke(width = stroke, cap = StrokeCap.Round),
            )

            drawArc(
                brush = Brush.sweepGradient(listOf(accentDim, accent, accent)),
                startAngle = startAngle,
                sweepAngle = sweepTotal * animated,
                useCenter = false,
                topLeft = topLeft,
                size = arcSize,
                style = Stroke(width = stroke, cap = StrokeCap.Round),
            )

            // The notch: a hard break in the track at the abstention threshold.
            val notchAngle = startAngle + sweepTotal * threshold.coerceIn(0f, 1f)
            val rad = Math.toRadians(notchAngle.toDouble())
            val cx = this.size.width / 2
            val cy = this.size.height / 2
            val radius = arcSize.width / 2
            val inner = radius - stroke * 0.95f
            val outer = radius + stroke * 0.95f
            drawLine(
                color = TeaColors.Base,
                start = androidx.compose.ui.geometry.Offset(
                    cx + (inner * cos(rad)).toFloat(), cy + (inner * sin(rad)).toFloat()
                ),
                end = androidx.compose.ui.geometry.Offset(
                    cx + (outer * cos(rad)).toFloat(), cy + (outer * sin(rad)).toFloat()
                ),
                strokeWidth = 5.dp.toPx(),
            )
            drawLine(
                color = TeaColors.TextLow,
                start = androidx.compose.ui.geometry.Offset(
                    cx + ((radius + stroke * 0.7f) * cos(rad)).toFloat(),
                    cy + ((radius + stroke * 0.7f) * sin(rad)).toFloat()
                ),
                end = androidx.compose.ui.geometry.Offset(
                    cx + ((radius + stroke * 1.5f) * cos(rad)).toFloat(),
                    cy + ((radius + stroke * 1.5f) * sin(rad)).toFloat()
                ),
                strokeWidth = 2.dp.toPx(),
            )
        }

        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(
                "%.0f".format(confidence * 100),
                style = MaterialTheme.typography.displayLarge.copy(fontSize = 44.sp),
                color = accent,
            )
            Text("PERCENT", style = MaterialTheme.typography.labelSmall, color = TeaColors.TextLow)
            Text(
                "threshold %.0f".format(threshold * 100),
                style = MaterialTheme.typography.labelSmall,
                color = TeaColors.TextLow,
            )
        }
    }
}

/** Camera framing brackets. Marks the capture area without boxing the subject in. */
@Composable
fun CornerBrackets(modifier: Modifier = Modifier, color: Color = TeaColors.Lime) {
    Canvas(modifier) {
        val len = size.minDimension * 0.13f
        val w = 3.dp.toPx()
        val r = 26.dp.toPx()
        val pad = 2.dp.toPx()

        fun corner(x: Float, y: Float, dx: Int, dy: Int) {
            drawLine(color, androidx.compose.ui.geometry.Offset(x + dx * r, y),
                androidx.compose.ui.geometry.Offset(x + dx * (r + len), y), w, StrokeCap.Round)
            drawLine(color, androidx.compose.ui.geometry.Offset(x, y + dy * r),
                androidx.compose.ui.geometry.Offset(x, y + dy * (r + len)), w, StrokeCap.Round)
        }
        corner(pad, pad, 1, 1)
        corner(size.width - pad, pad, -1, 1)
        corner(pad, size.height - pad, 1, -1)
        corner(size.width - pad, size.height - pad, -1, -1)
    }
}

/** Translucent panel with a hairline edge. */
@Composable
fun GlassCard(
    modifier: Modifier = Modifier,
    radius: Dp = TeaShape.card,
    content: @Composable () -> Unit,
) {
    Box(
        modifier
            .clip(RoundedCornerShape(radius))
            .background(TeaColors.Surface.copy(alpha = 0.86f))
            .border(1.dp, TeaColors.Outline, RoundedCornerShape(radius))
            .padding(18.dp)
    ) { content() }
}

/** One row of the alternatives list: what else the model considered, and how strongly. */
@Composable
fun ProbabilityBar(
    label: String,
    probability: Float,
    highlighted: Boolean,
    modifier: Modifier = Modifier,
) {
    val accent = if (highlighted) TeaColors.Lime else TeaColors.TextLow
    Column(modifier.fillMaxWidth()) {
        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                label,
                style = MaterialTheme.typography.titleMedium.copy(
                    fontWeight = if (highlighted) FontWeight.Bold else FontWeight.Normal
                ),
                color = if (highlighted) TeaColors.TextHigh else TeaColors.TextMid,
            )
            Text(
                "%.1f%%".format(probability * 100),
                style = MaterialTheme.typography.labelLarge,
                color = accent,
            )
        }
        Box(
            Modifier
                .fillMaxWidth()
                .height(6.dp)
                .padding(top = 6.dp)
                .clip(RoundedCornerShape(TeaShape.pill))
                .background(TeaColors.Outline)
        ) {
            Box(
                Modifier
                    .fillMaxWidth(probability.coerceIn(0f, 1f))
                    .fillMaxSize()
                    .clip(RoundedCornerShape(TeaShape.pill))
                    .background(accent)
            )
        }
    }
}

/** Small monospace tag for a measured value. */
@Composable
fun ReadingChip(label: String, value: String, modifier: Modifier = Modifier) {
    Row(
        modifier
            .clip(RoundedCornerShape(TeaShape.chip))
            .background(TeaColors.SurfaceHigh)
            .padding(horizontal = 12.dp, vertical = 7.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Text(label, style = MaterialTheme.typography.labelSmall, color = TeaColors.TextLow)
        Text(value, style = MaterialTheme.typography.labelLarge, color = TeaColors.TextHigh)
    }
}
