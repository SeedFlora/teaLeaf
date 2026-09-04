# Field Log Schema — TeaLeaf AI

The application records every inference to an append-only JSONL log alongside the captured
image, so that a session in the field can later be pulled over ADB and analyzed.

## 1. The organizing principle

Each record separates three kinds of information that must never be conflated:

| Block | Contains | Depends on human judgment? |
|---|---|---|
| `measurement` | probabilities, latency, thermal state, model identity | **No** — produced entirely by the device |
| `annotation` | the user's verdict on whether the prediction was right | **Yes** — a human opinion of stated expertise |
| `expert_verification` | a later confirmation by a qualified specialist | **Yes**, but by a qualified party |

This separation is structural, not cosmetic. It means the analysis script can compute the
confidence distribution, abstention rate, and latency profile under real field conditions
**without reading a single label** — so those results stand regardless of who annotated. Only
the label-dependent analyses are restricted to the annotated subset, and accuracy claims are
restricted further, to the `expert_verification` subset.

`expert_verification` exists and is empty from the first capture on purpose. If a qualified
person later verifies part of the set, that subset is analyzable immediately, with no
retro-fitting of a format that field data has already been collected in.

## 2. What this data can and cannot support

**Supported without any label:**
- Confidence distribution on field images versus the studio-captured test partition.
- Abstention rate under real capture conditions — does the model decline when it should?
- End-to-end camera-to-result latency on the physical device.
- Failure characterization: what the model does on out-of-distribution input.

**Supported only on the annotated subset, and only as non-expert field annotation:**
- Apparent agreement rate between the model and a non-specialist observer.
- Qualitative error patterns worth investigating.

**Supported only on the expert-verified subset:**
- Any accuracy, precision, recall, or F1 figure.

**Never supported:**
- Reporting non-expert agreement as external validation accuracy.
- Treating a single annotator's verdict as ground truth for a pathology whose classes are
  genuinely difficult to distinguish visually — gray blight versus brown blight in particular.

## 3. Record schema

One JSON object per line in `field_log.jsonl`.

```jsonc
{
  "schema_version": "1.0",
  "record_id": "8f14e45f-ceea-467a-9d0b-2c1e8f3a91bd",
  "session_id": "2b0c9a17-...",              // one per app launch
  "capture_index": 7,                         // position within the session
  "timestamp_utc": "2026-07-21T02:14:33.512Z",
  "timezone_offset_minutes": 420,

  "image": {
    "file": "images/img_20260721_021433_007.jpg",
    "sha256": "…",                            // ties the record to the exact pixels
    "width": 3000, "height": 4000,
    "source": "camera",                       // camera | gallery
    "camera_facing": "back",
    "exif_orientation": 1,
    "rotation_applied_degrees": 90            // what the app did to make it upright
  },

  // ---- MEASURED: no human involvement ----
  "measurement": {
    "model": {
      "file": "tealeaf_int8.tflite",
      "sha256": "…",
      "variant": "int8",                      // fp32 | fp16 | int8
      "version": "1.0.0",
      "input_size": 224,
      "input_convention": "raw [0,255] float32; normalization embedded in the graph",
      "class_order": ["tea_algal_leaf_spot","brown_blight","gray_blight",
                      "helopeltis_damage","red_spider_mite_damage",
                      "green_mirid_bug_damage","healthy_leaf"]
    },
    "probabilities_raw": [0.02,0.71,0.19,0.03,0.02,0.02,0.01],
    "probabilities_calibrated": [0.05,0.58,0.24,0.05,0.04,0.03,0.01],
    "temperature": 1.37,                      // fitted on validation only
    "top1_index": 1,
    "top1_confidence_calibrated": 0.58,
    "top3_indices": [1,2,0],
    "predictive_entropy": 1.31,
    "abstained": true,
    "abstention_threshold": 0.62,             // chosen by validation risk-coverage
    "abstention_reason": "below_confidence_threshold",

    "latency_ms": {
      "preprocess": 8.4,                      // decode, rotate, resize to 224
      "inference": 21.7,                      // pure interpreter run
      "postprocess": 0.3,
      "end_to_end": 30.4                      // shutter to displayed result
    },
    "runtime": {
      "delegate": "xnnpack",                  // xnnpack | gpu | cpu
      "num_threads": 4
    },
    "device_state": {
      "thermal_status": 0,                    // PowerManager thermal status at capture
      "battery_percent": 51,
      "is_charging": false,
      "available_ram_mb": 1802
    }
  },

  // ---- HUMAN JUDGEMENT: filled by the user, possibly later ----
  "annotation": {
    "annotator_id": "annotator-01",           // pseudonymous, no personal data
    "annotator_expertise": "none",            // none | farmer | extension_officer |
                                              // agronomist | plant_pathologist
    "verdict": "incorrect",                   // correct | incorrect | unsure | null
    "claimed_true_class": "gray_blight",      // null unless verdict is "incorrect"
    "annotator_confidence": "medium",         // low | medium | high | null
    "annotated_at_utc": "2026-07-21T02:15:02.004Z",
    "notes": "lesion edge was hard to see in shade"
  },

  // ---- EXPERT VERIFICATION: empty until a qualified party reviews ----
  "expert_verification": {
    "verified": false,
    "expert_role": null,                      // e.g. "agricultural extension officer"
    "expert_class": null,
    "verified_at_utc": null,
    "agrees_with_annotator": null
  },

  // ---- CAPTURE CONTEXT: self-reported, all optional ----
  "context": {
    "lighting": "shade",                      // bright | overcast | shade | low
    "leaf_attached_to_plant": true,           // the studio-vs-field distinction
    "background": "field",                    // field | plain | mixed
    "location_label": "Kebun Teh A",          // free text; NOT coordinates
    "gps_recorded": false                     // GPS is never collected by default
  },

  "consent": {
    "research_use_granted": true,
    "granted_at_utc": "2026-07-21T02:10:00.000Z"
  }
}
```

## 4. Privacy constraints

- The log and images are written to app-specific storage only. No other app directory, no
  camera roll, no personal media is read or written.
- **No GPS coordinates are collected.** `location_label` is optional free text the user types.
- No network permission is requested, so nothing can leave the device by any path other than
  the user physically connecting a cable.
- `annotator_id` is a pseudonym generated on device; no name, email, or account is recorded.
- A consent flag is recorded per session, and captures without granted consent are excluded
  from analysis rather than silently used.

## 5. Retrieval

Written under the app's external files directory, which is readable by `adb pull` on a
non-rooted device without granting any storage permission:

```
/sdcard/Android/data/<applicationId>/files/research/
    field_log.jsonl
    images/
    session_manifest.json
```

Pulled with `src/benchmarking/pull_field_log.py`, which copies the directory, verifies every
image against its recorded SHA-256, and refuses to merge records whose image hash does not
match — so a truncated or corrupted transfer fails loudly instead of quietly changing results.

## 6. Analysis entry points

| Question | Requires labels? | Subset used |
|---|---|---|
| Confidence distribution, field vs studio test set | No | all consented records |
| Abstention rate under field conditions | No | all consented records |
| End-to-end and pure-inference latency | No | all consented records |
| Model/annotator agreement rate | Yes | `annotation.verdict != null` |
| Accuracy, precision, recall, F1 | Yes | `expert_verification.verified == true` |

The analysis script computes each group separately and labels every output with the subset and
the annotator expertise it rests on, so a figure can never be promoted to a stronger claim than
its evidence supports.
