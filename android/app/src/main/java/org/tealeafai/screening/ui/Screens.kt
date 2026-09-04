package org.tealeafai.screening.ui

import android.graphics.Bitmap
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import org.tealeafai.screening.ml.TeaLeafClassifier

/** Everything the result screen needs, assembled once so the UI stays a pure function of it. */
data class ScreeningResult(
    val photo: Bitmap,
    val heatmap: Bitmap?,
    val prediction: TeaLeafClassifier.Prediction,
    val labelsEn: List<String>,
    val labelsId: List<String>,
    val threshold: Float,
    val modelVariant: String,
    val modelVersion: String,
    val peripheralMass: Float,
)

@Composable
fun ResultScreen(
    result: ScreeningResult,
    onRetake: () -> Unit,
    onFeedback: (verdict: String, claimedClass: String?) -> Unit,
    modifier: Modifier = Modifier,
) {
    var showHeatmap by remember { mutableStateOf(true) }
    var feedbackGiven by remember { mutableStateOf<String?>(null) }
    var correcting by remember { mutableStateOf(false) }

    val p = result.prediction
    val abstained = p.abstained
    val accent = if (abstained) TeaColors.Amber else TeaColors.Lime
    val topEn = result.labelsEn.getOrElse(p.topIndex) { "Unknown" }
    val topId = result.labelsId.getOrElse(p.topIndex) { "" }

    Column(
        modifier
            .fillMaxSize()
            .background(TeaColors.Base)
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 18.dp)
            .padding(bottom = 32.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Spacer(Modifier.height(14.dp))

        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text("Reading", style = MaterialTheme.typography.headlineMedium, color = TeaColors.TextHigh)
            ReadingChip("VARIANT", result.modelVariant.uppercase())
        }

        // --- capture with the activation overlay -------------------------------------
        Box(
            Modifier
                .fillMaxWidth()
                .aspectRatio(1f)
                .clip(RoundedCornerShape(TeaShape.card))
                .background(TeaColors.Surface)
        ) {
            val shown = if (showHeatmap && result.heatmap != null) result.heatmap else result.photo
            Image(
                bitmap = shown.asImageBitmap(),
                contentDescription = if (showHeatmap)
                    "Captured leaf with the model's evidence regions highlighted"
                else "Captured leaf",
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Crop,
            )
            CornerBrackets(Modifier.fillMaxSize().padding(12.dp), accent.copy(alpha = 0.55f))

            if (result.heatmap != null) {
                Row(
                    Modifier
                        .align(Alignment.BottomStart)
                        .padding(14.dp)
                        .clip(RoundedCornerShape(TeaShape.pill))
                        .background(TeaColors.Scrim)
                        .clickable { showHeatmap = !showHeatmap }
                        .padding(horizontal = 14.dp, vertical = 9.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    Box(
                        Modifier
                            .size(9.dp)
                            .clip(CircleShape)
                            .background(if (showHeatmap) accent else TeaColors.TextLow)
                    )
                    Text(
                        if (showHeatmap) "EVIDENCE ON" else "EVIDENCE OFF",
                        style = MaterialTheme.typography.labelSmall,
                        color = TeaColors.TextHigh,
                    )
                }
            }
        }

        // --- the reading -------------------------------------------------------------
        GlassCard {
            Column(
                Modifier.fillMaxWidth(),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(14.dp),
            ) {
                ConfidenceRing(
                    confidence = p.topConfidence,
                    threshold = result.threshold,
                    abstained = abstained,
                )

                if (abstained) {
                    Text(
                        "Not confident enough to name a condition",
                        style = MaterialTheme.typography.titleLarge,
                        color = TeaColors.Amber,
                    )
                    Text(
                        "Retake in brighter light with one leaf filling most of the frame, " +
                            "or ask a qualified agricultural specialist. The closest match was " +
                            "$topEn, below the confidence this model needs to answer.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = TeaColors.TextMid,
                    )
                } else {
                    Text(topEn, style = MaterialTheme.typography.headlineMedium, color = TeaColors.TextHigh)
                    if (topId.isNotEmpty()) {
                        Text(topId, style = MaterialTheme.typography.bodyMedium, color = TeaColors.TextMid)
                    }
                }
            }
        }

        // --- what else it considered --------------------------------------------------
        GlassCard {
            Column(verticalArrangement = Arrangement.spacedBy(14.dp)) {
                Text("Also considered", style = MaterialTheme.typography.titleLarge, color = TeaColors.TextHigh)
                p.top3Indices.forEachIndexed { rank, idx ->
                    ProbabilityBar(
                        label = result.labelsEn.getOrElse(idx) { "-" },
                        probability = p.probabilitiesCalibrated.getOrElse(idx) { 0f },
                        highlighted = rank == 0 && !abstained,
                    )
                }
            }
        }

        // --- how to read the overlay --------------------------------------------------
        if (result.heatmap != null) {
            GlassCard {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("About the highlight", style = MaterialTheme.typography.titleLarge, color = TeaColors.TextHigh)
                    Text(
                        "Bright areas are where the model's evidence for this class is " +
                            "concentrated. They show WHERE it looked, not WHY a leaf is " +
                            "affected, and the model does not reason about lesion biology.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = TeaColors.TextMid,
                    )
                    if (result.peripheralMass > 0.5f) {
                        Text(
                            "Most of the evidence sits near the frame edges rather than on the " +
                                "leaf. Treat this reading with extra caution and retake with the " +
                                "leaf centred.",
                            style = MaterialTheme.typography.bodyMedium,
                            color = TeaColors.Amber,
                        )
                    }
                }
            }
        }

        // --- measured facts ------------------------------------------------------------
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
            ReadingChip("INFER", "%.1f ms".format(p.inferenceMs), Modifier.weight(1f))
            ReadingChip("TOTAL", "%.1f ms".format(p.preprocessMs + p.inferenceMs + p.postprocessMs), Modifier.weight(1f))
            ReadingChip("ENTROPY", "%.2f".format(p.entropy), Modifier.weight(1f))
        }

        // --- field feedback -------------------------------------------------------------
        GlassCard {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Text("Was this right?", style = MaterialTheme.typography.titleLarge, color = TeaColors.TextHigh)
                Text(
                    "Your answer is stored on this phone with the reading, so the model can be " +
                        "checked against real captures later. It is recorded as your judgement, " +
                        "kept separate from the model's own output.",
                    style = MaterialTheme.typography.bodySmall,
                    color = TeaColors.TextLow,
                )

                if (feedbackGiven == null) {
                    Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                        VerdictButton("Correct", TeaColors.Lime, Modifier.weight(1f)) {
                            feedbackGiven = "correct"; onFeedback("correct", null)
                        }
                        VerdictButton("Wrong", TeaColors.Danger, Modifier.weight(1f)) {
                            correcting = true
                        }
                        VerdictButton("Not sure", TeaColors.TextLow, Modifier.weight(1f)) {
                            feedbackGiven = "unsure"; onFeedback("unsure", null)
                        }
                    }
                } else {
                    Text(
                        "Recorded: $feedbackGiven",
                        style = MaterialTheme.typography.labelLarge,
                        color = TeaColors.Lime,
                    )
                }

                AnimatedVisibility(visible = correcting && feedbackGiven == null) {
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text(
                            "Which condition is it?",
                            style = MaterialTheme.typography.titleMedium,
                            color = TeaColors.TextMid,
                        )
                        result.labelsEn.forEachIndexed { i, name ->
                            Text(
                                name,
                                style = MaterialTheme.typography.bodyMedium,
                                color = TeaColors.TextHigh,
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .clip(RoundedCornerShape(TeaShape.chip))
                                    .background(TeaColors.SurfaceHigh)
                                    .clickable {
                                        feedbackGiven = "incorrect"
                                        correcting = false
                                        onFeedback("incorrect", result.labelsEn[i])
                                    }
                                    .padding(horizontal = 14.dp, vertical = 11.dp),
                            )
                        }
                    }
                }
            }
        }

        Text(
            "AI screening aid, not a diagnosis. It does not replace a qualified agricultural " +
                "specialist and gives no treatment advice.",
            style = MaterialTheme.typography.bodySmall,
            color = TeaColors.TextLow,
        )

        Box(
            Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(TeaShape.pill))
                .background(TeaColors.Lime)
                .clickable { onRetake() }
                .padding(vertical = 17.dp),
            contentAlignment = Alignment.Center,
        ) {
            Text(
                "Scan another leaf",
                style = MaterialTheme.typography.titleLarge,
                color = TeaColors.Base,
                fontWeight = FontWeight.Bold,
            )
        }
    }
}

@Composable
private fun VerdictButton(
    label: String,
    accent: Color,
    modifier: Modifier = Modifier,
    onClick: () -> Unit,
) {
    Box(
        modifier
            .clip(RoundedCornerShape(TeaShape.pill))
            .background(TeaColors.SurfaceHigh)
            .border(1.dp, accent.copy(alpha = 0.5f), RoundedCornerShape(TeaShape.pill))
            .clickable { onClick() }
            .padding(vertical = 13.dp),
        contentAlignment = Alignment.Center,
    ) {
        Text(label, style = MaterialTheme.typography.titleMedium, color = accent)
    }
}

/** Capture screen shown before a reading exists. */
@Composable
fun CaptureScreen(
    modelReady: Boolean,
    loadError: String?,
    busy: Boolean,
    onCapture: () -> Unit,
    onPickImage: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Column(
        modifier
            .fillMaxSize()
            .background(TeaColors.Base)
            .padding(horizontal = 18.dp),
        verticalArrangement = Arrangement.spacedBy(18.dp),
    ) {
        Spacer(Modifier.height(20.dp))
        Text("TeaLeaf AI", style = MaterialTheme.typography.displayLarge, color = TeaColors.TextHigh)
        Text(
            "Point at one leaf, fill the frame, keep it in even light.",
            style = MaterialTheme.typography.bodyMedium,
            color = TeaColors.TextMid,
        )

        Box(
            Modifier
                .fillMaxWidth()
                .weight(1f)
                .clip(RoundedCornerShape(TeaShape.card))
                .background(TeaColors.Surface)
                .border(1.dp, TeaColors.Outline, RoundedCornerShape(TeaShape.card)),
            contentAlignment = Alignment.Center,
        ) {
            CornerBrackets(Modifier.fillMaxSize().padding(20.dp))
            when {
                loadError != null -> Column(
                    Modifier.padding(28.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(10.dp),
                ) {
                    Text("Model did not load", style = MaterialTheme.typography.titleLarge, color = TeaColors.Danger)
                    Text(loadError, style = MaterialTheme.typography.bodySmall, color = TeaColors.TextMid)
                }
                busy -> Text("Reading…", style = MaterialTheme.typography.titleLarge, color = TeaColors.Lime)
                else -> Text(
                    "Camera preview",
                    style = MaterialTheme.typography.labelLarge,
                    color = TeaColors.TextLow,
                )
            }
        }

        Row(
            Modifier
                .fillMaxWidth()
                .padding(bottom = 28.dp),
            horizontalArrangement = Arrangement.spacedBy(14.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Box(
                Modifier
                    .weight(1f)
                    .clip(RoundedCornerShape(TeaShape.pill))
                    .background(TeaColors.SurfaceHigh)
                    .clickable(enabled = modelReady && !busy) { onPickImage() }
                    .padding(vertical = 17.dp),
                contentAlignment = Alignment.Center,
            ) {
                Text("From gallery", style = MaterialTheme.typography.titleMedium, color = TeaColors.TextHigh)
            }
            Box(
                Modifier
                    .size(76.dp)
                    .clip(CircleShape)
                    .background(if (modelReady && !busy) TeaColors.Lime else TeaColors.LimeDim)
                    .clickable(enabled = modelReady && !busy) { onCapture() },
                contentAlignment = Alignment.Center,
            ) {
                Box(
                    Modifier
                        .size(26.dp)
                        .clip(CircleShape)
                        .background(TeaColors.Base)
                )
            }
        }
    }
}
