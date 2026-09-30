// Builds docs/CSET485_ROAD_SHIELD_Milestone3_Report_Draft.docx.
// Every number below is copied from a result file in the repository (model
// cards, benchmark and quantization reports); see docs/REPORT_ALIGNMENT.md.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType,
  AlignmentType, HeadingLevel, ShadingType, LevelFormat, BorderStyle, PageNumber,
  Footer, Header, TableOfContents, PageBreak,
} = require("docx");

const FONT = "Calibri";
const CONTENT_WIDTH = 9026; // A4 with 1" margins, in DXA
const PENDING = "PENDING";

// Results are read from the files the training and benchmark scripts write, so
// the report cannot drift from the models. Missing files leave PENDING markers.
const ROOT = path.resolve(__dirname, "..", "..");
const readJSON = (rel) => { try { return JSON.parse(fs.readFileSync(path.join(ROOT, rel), "utf8")); } catch { return null; } };
const RD = readJSON("checkpoints/detectors/road_damage.json");
const QUANT = readJSON("checkpoints/quantization_report.json") || {};
const LAT = readJSON("checkpoints/perception_latency.json");
const f3 = (x) => (x == null ? "—" : Number(x).toFixed(3));
const rdTest = RD ? RD.metrics.test : null;
const rdEpochs = RD ? (RD.epochs_trained || RD.config.epochs) : null;
const rdClass = (name) => {
  const m = rdTest && rdTest.per_class[name];
  if (!m) return [PENDING, PENDING, PENDING, PENDING];
  // With no correct detections precision is undefined; the evaluator reports 1.0, which would mislead.
  return [m.recall === 0 ? "— (no detections)" : f3(m.precision), f3(m.recall), f3(m.mAP50), f3(m.mAP50_95)];
};

// "**bold** plain" -> runs. A PENDING token is highlighted so it cannot be missed.
function runs(text, base = {}) {
  const out = [];
  text.split(/(\*\*[^*]+\*\*)/).forEach((part) => {
    if (!part) return;
    const bold = part.startsWith("**") && part.endsWith("**");
    const body = bold ? part.slice(2, -2) : part;
    body.split(/(PENDING[^.;|]*)/).forEach((piece) => {
      if (!piece) return;
      const pending = piece.startsWith(PENDING);
      out.push(new TextRun({
        text: piece, font: FONT, size: base.size || 22, bold: bold || base.bold,
        italics: base.italics || pending, color: pending ? "B45309" : base.color,
        highlight: pending ? "yellow" : undefined,
      }));
    });
  });
  return out;
}

const p = (text, opts = {}) => new Paragraph({
  children: runs(text, opts), spacing: { after: 120, line: 276 },
  alignment: opts.align || AlignmentType.JUSTIFIED,
});
const h1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, keepNext: true, keepLines: true, children: [new TextRun({ text, font: FONT })],
  spacing: { before: 280, after: 140 } });
const h2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, keepNext: true, keepLines: true, children: [new TextRun({ text, font: FONT })],
  spacing: { before: 200, after: 100 } });
const bullet = (text) => new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: runs(text),
  spacing: { after: 60 } });
const caption = (text) => new Paragraph({ children: runs(text, { italics: true, size: 18, color: "555555" }),
  spacing: { before: 60, after: 200 } });

function table(headers, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const scale = CONTENT_WIDTH / total;
  const w = widths.map((x) => Math.round(x * scale));
  w[w.length - 1] += CONTENT_WIDTH - w.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const cell = (text, i, header, keep = true) => new TableCell({
    width: { size: w[i], type: WidthType.DXA }, borders,
    shading: header ? { type: ShadingType.CLEAR, color: "auto", fill: "1F3864" } : undefined,
    margins: { top: 60, bottom: 60, left: 90, right: 90 },
    children: [new Paragraph({ keepNext: keep, children: runs(String(text), { size: 18, bold: header, color: header ? "FFFFFF" : undefined }) })],
  });
  return new Table({
    width: { size: CONTENT_WIDTH, type: WidthType.DXA }, columnWidths: w,
    rows: [
      new TableRow({ tableHeader: true, cantSplit: true, children: headers.map((t, i) => cell(t, i, true)) }),
      ...rows.map((r, n) => new TableRow({ cantSplit: true, children: r.map((t, i) => cell(t, i, false, rows.length <= 8 && n < rows.length - 1)) })),
    ],
  });
}

// ------------------------------------------------------------------ content
const title = [
  new Paragraph({ spacing: { before: 1800, after: 200 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "ROAD-SHIELD", font: FONT, size: 56, bold: true, color: "1F3864" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 400 },
    children: [new TextRun({ text: "An AI-Powered Mobile Sensing Framework for Multi-Modal Road Distress Detection and Urban Safety Monitoring Using Public Transport Fleets", font: FONT, size: 28 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 },
    children: [new TextRun({ text: "Milestone 3 Progress Report (Draft)", font: FONT, size: 32, bold: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 },
    children: [new TextRun({ text: "CSET485 — AI and Society  ·  Smart India Hackathon 2026, SIH26124 (Bharat Electronics Limited)", font: FONT, size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: runs(RD ? "Draft of 1 October 2026. Highlighted PENDING items are filled in once the licence-plate detector finishes training." : "Draft prepared 30 September 2026. Highlighted PENDING items are filled in once the road-damage and licence-plate detectors finish training.", { italics: true, size: 20 }) }),
  new Paragraph({ children: [new PageBreak()] }),
];

const meta = [
  h1("Project Metadata and Team Information"),
  table(["Field", "Detail"], [
    ["Course", "CSET485 — AI and Society"],
    ["Project type", "Academic Mini Project, Milestone 3 submission"],
    ["Hackathon alignment", "Smart India Hackathon 2026, Problem Statement SIH26124 (Bharat Electronics Limited)"],
    ["Theme", "Smart Automation, Intelligent Transportation Systems, Clean Mobility"],
    ["Team", "Team ID 42"],
    ["Code", "github.com/udbhav968-creator/SIH_PROJECT, branch feature/road-scene-perception"],
  ], [2, 6]),
  caption("Table 1. Project metadata."),
  table(["S.No.", "Student", "Roll Number", "Branch"], [
    ["1", "Udbhav (Team Lead)", "S24CSEU0095", "Computer Science & Engineering"],
    ["2", "Yuvraj Pathak", "S24CSEU1182", "Computer Science & Engineering"],
    ["3", "Naman Bhandari", "S24CSEU0611", "Computer Science & Engineering"],
    ["4", "Prakhar", "S24CSEU1171", "Computer Science & Engineering"],
    ["5", "Amarjeet Kumar", "S24CSEU1968", "Computer Science & Engineering"],
  ], [1, 3, 2.4, 3.6]),
  caption("Table 2. Student authors."),
];

const abstract = [
  h1("Abstract"),
  p("Milestone 2 described ROAD-SHIELD, a system that turns city buses into a road-inspection network: a dashcam and an accelerometer on each bus, an edge computer that finds and measures road defects, and a central ledger that deduplicates reports and issues tamper-evident repair orders. Milestone 3 set out to move the system from photographs of single defects to full dashcam scenes on Indian roads, to implement the privacy and edge-transmission design stated in Milestone 2, and to measure every claim against held-out data."),
  p("This milestone adds a scene-perception layer of three separately trained YOLO detectors — traffic participants (COCO), road damage (RDD2022, Indian subset) and road markings (CDSet-3434) — served on the CPU through ONNX Runtime. The zebra-crossing detector reaches **test mAP50 0.869** and, asked whether a frame contains a crossing, is correct **every time it says yes (0 false alarms on 263 crossing-free frames)** while finding 75.4% of crossings — 3.2 times more than the geometric method it replaces. We implemented the Milestone 2 design items that existed only on paper: DPDP privacy redaction, sub-kilobyte signed edge packets, the Butterworth IMU filter (78.7% accuracy under engine vibration versus 60.1% without it), a calibrated frame-quality gate, the fairness-constrained repair Priority Index and INT8 quantization (crossing detector 1.3× faster and 2.4× smaller for a 0.006 mAP50 cost)."),
  p("Measuring the Milestone 2 claims also corrected eleven of them, listed in Section 6. The software is now covered by 178 automated tests run on every commit; two security defects in the web API were found and fixed. " + (RD ? `The road-damage detector, trained on a laptop CPU, reaches test mAP50 ${f3(rdTest.mAP50)}; Section 5.2 reports it class by class, including the class it does not yet detect.` : "Road-damage detection results are PENDING the completion of training.")),
];

const progress = [
  h1("1. Progress Against the Milestone 3 Plan"),
  p("Milestone 2 closed with a four-item plan. Table 3 reports each item and the additional work that measurement showed was necessary."),
  table(["Milestone 2 plan item", "Status", "Outcome"], [
    ["1. Ingest RDD2022 Indian roads", "Done", "7,706 labelled Indian images downloaded and prepared (5,368 train / 1,172 val / 1,166 test); " + (RD ? `road-damage detector trained: test mAP50 ${f3(rdTest.mAP50)} (CPU run; see Section 5.2)` : "road-damage detector PENDING training completion")],
    ["2. Real bus sensor mounting", "Not started", "Requires hardware access; the IMU model now trains on real field-recorded vehicle logs (Section 3)"],
    ["3. INT8 model quantization", "Done (CPU)", "Crossing detector 61 → 48 ms, 10.0 → 4.1 MB, mAP50 0.864 → 0.858; Raspberry Pi / Jetson benchmarks not yet run"],
    ["4. Live command-centre connection", "Partial", "Ranked repair ledger and scene-detection page served by the API; no live bus telemetry feed yet"],
    ["Added: zebra-crossing detection", "Done", "Test mAP50 0.869; see Section 5.1"],
    ["Added: DPDP redaction, edge packets, IMU filter, frame gate, Priority Index", "Done", "Stated in Milestone 2 but not implemented; now implemented and tested"],
    ["Added: licence-plate detector for redaction", "Ready to train", "Dataset and configuration prepared; model PENDING training"],
  ], [3.2, 1.3, 5.5]),
  caption("Table 3. Milestone 3 plan items and outcomes."),
];

const architecture = [
  h1("2. System Architecture"),
  p("ROAD-SHIELD now has two analysis paths behind one server. The **inspection pipeline** from Milestone 2 turns a single road photograph into a classified, measured and costed repair record. The new **scene-perception layer** analyses a whole dashcam frame and finds everything in it at once."),
  h2("2.1 Scene perception"),
  bullet("**Frame-quality gate.** Rejects frames with a covered lens, darkness, glare or heavy defocus before any detector runs, saving edge CPU (Section 4.2)."),
  bullet("**Three detectors, kept separate on purpose.** Each training set labels only its own classes; a single merged model would learn that every unlabelled pothole in a crossing photograph is background. Separate models can be retrained, versioned and rolled back independently."),
  bullet("**Traffic:** COCO-pretrained YOLO — pedestrians, cars, buses, trucks, two-wheelers, bicycles, traffic lights, stop signs and stray animals."),
  bullet("**Road damage:** YOLO11n trained on RDD2022 India — longitudinal (D00), transverse (D10) and alligator (D20) cracks and potholes (D40)."),
  bullet("**Road markings:** YOLO11n trained on CDSet-3434 — zebra crossings and lane guide arrows."),
  bullet("**Scene facts and alerts.** Geometric rules over the detected boxes: road users standing on a crossing (critical alert), potholes in the bus's path, vehicles blocking a crossing. Every fact states the rule that produced it."),
  bullet("**Privacy head.** A licence-plate detector used only to blur plates; plate boxes never appear in results, APIs or transmitted packets."),
  h2("2.2 From bus to ledger"),
  p("For each analysed frame the bus sends one signed JSON packet of at most 1,024 bytes (Section 4.5). The server verifies the signature, deduplicates reports within 8 m using the Haversine distance, and orders the repair queue by the Priority Index (Section 4.6). Work orders keep the Milestone 2 SHA-256 seal."),
];

const data = [
  h1("3. Datasets and Data Hygiene"),
  table(["Dataset", "Use", "Size used", "Licence", "Hygiene applied"], [
    ["RDD2022, India subset (Arya et al.)", "Road-damage detection", "7,706 images; 3,550 training boxes", "CC BY-SA 4.0", "Clean-road images capped at one third of the training list; transverse cracks are rare (50 boxes), reported per class"],
    ["CDSet-3434 (Zhang et al.)", "Zebra crossings, guide arrows", "3,388 frames kept of 4,850", "Apache-2.0", "Official split is leaky (Section 3.1); re-split by time block with a 30-frame purge margin"],
    ["Vehicle Registration Plates (Roboflow)", "Plate redaction", "8,823 images", "CC BY 4.0", "3,904 synthetic rendered plates (44%) used for training only; val/test are real photographs"],
    ["DNIT highway corpus (Brazil)", "Distress classifier, segmenter", "1,921 crack and 564 pothole polygons", "Public", "Split by source photograph (unchanged from Milestone 2)"],
    ["Pothole-Project IMU logs (India)", "Shock classifier", "852 windows (688 train / 164 held-out)", "Public repository", "Real field recordings; time-block split with guard gap"],
  ], [2.2, 1.6, 1.8, 1.2, 3.2]),
  caption("Table 4. Datasets used in Milestone 3."),
  h2("3.1 A leakage finding in CDSet"),
  p("All 3,434 labelled CDSet frames come from only three dashcam videos, and **every frame in the official test split is one or two frames away from a training frame**. Adjacent video frames are nearly identical, so any score on the official split measures memorisation. We re-split by contiguous 300-frame time blocks per video, discarded 524 frames within 30 frames of a split boundary, and kept 938 crossing-free frames out of training so they could measure false alarms. All results in Section 5.1 use this split."),
  h2("3.2 Synthetic data in the plate dataset"),
  p("44% of the licence-plate images are computer-rendered plates. They help a detector learn to localise plates, but they are easy; a test score that included them would overstate real-world performance. They are therefore confined to the training split."),
];

const methods = [
  h1("4. Methods"),
  h2("4.1 Detector training protocol"),
  p("Every shipped detector is produced by one script (training/train_detector.py) from one configuration file, so a weight file traces back to its data and settings. Training starts from COCO-pretrained YOLO11n. On a laptop CPU the learning-rate schedule is fitted to a wall-clock budget; on a GPU a fixed epoch count is used. After training, the script: (1) chooses a per-class confidence threshold at the F1 peak on the **validation** split; (2) scores box mAP on the **test** split, which the thresholds never saw; (3) scores image-level questions (\"does this frame contain a crossing?\") and the false-alarm rate on clean frames; (4) exports ONNX and writes a model card with the SHA-256 of each file. An interrupted run resumes from its last completed epoch."),
  h2("4.2 Frame-quality gate"),
  p("Milestone 2 proposed skipping frames whose Laplacian variance is below 42.5. Measured on 450 real road frames from three camera sources, that threshold would have **discarded 10.4% of good frames**, because texture varies ten-fold between cameras (median 1,192 on CDSet dashcam frames, 127 on the project photographs). The gate instead combines three calibrated checks on the road region: Laplacian variance below 5.0, mean luminance below 15, or more than half the pixels saturated."),
  h2("4.3 IMU band-pass filter"),
  p("Accelerometer windows pass through a 4th-order Butterworth band-pass filter (0.5–25 Hz, zero phase) as the first step of the saved model, so training and deployment cannot disagree. We adopted it only after measuring it (Section 5.3)."),
  h2("4.4 Privacy redaction (DPDP Act 2023)"),
  p("Every detected person is blurred before an image leaves the system; images produced by the server and command-line tools are redacted by default. Plates are blurred by the dedicated plate detector at a recall-leaning threshold of 0.15, because a missed plate costs far more than blurring a false positive. Until that model is trained the output states explicitly that plates were not redacted."),
  h2("4.5 Edge event packet"),
  p("A bus transmits, per analysed frame, a JSON packet of at most 1,024 bytes: road findings with class code, confidence and normalised box; counts (never boxes) of people and vehicles; and alerts. When a crowded frame would exceed the limit, the least important findings are dropped first (potholes are kept longest) and the number dropped is recorded. Packets carry an HMAC-SHA256 keyed per bus: unlike a bare SHA-256, it cannot be recomputed by someone who alters the packet."),
  h2("4.6 Repair Priority Index"),
  p("The Milestone 2 index PI = w1(100 − PCI) + w2·Vol + w3·Traffic is implemented with each term scaled to [0, 1] so that the weights mean what they say (defaults 0.5, 0.2, 0.3):"),
  p("**PI = 100 · [ 0.5·(100 − PCI)/100 + 0.2·V/(V + 0.06) + 0.3·log(1 + AADT)/log(1 + 100,000) ]**", { align: AlignmentType.CENTER }),
  p("Volume saturates (a 2 m³ crater is not forty times more urgent than a small pothole) and traffic is logarithmic. The function accepts only measured road quantities; it has no input for ward, locality or number of complaints, and an automated test confirms that two defects with identical measurements in a VIP area and an outer colony receive identical priority. When traffic has not been measured, the traffic term is dropped and the weights renormalised rather than guessed."),
  h2("4.7 INT8 quantization"),
  p("Models are quantized with static, per-channel INT8 quantization calibrated on training images only and scored against the full-precision model on the same held-out images. For YOLO detectors the final detection block is kept in full precision: it concatenates box coordinates (0–416 pixels) with class probabilities (0–1), and a single INT8 scale for both collapsed the crossing detector's mAP50 from 0.864 to 0.100 in our first attempt."),
];

const results = [
  h1("5. Results"),
  p("All figures below are measured on held-out data that the models and thresholds never saw. Hardware: Intel Core i5-1145G7 laptop CPU, no GPU."),
  h2("5.1 Zebra-crossing detector"),
  table(["Class", "Precision", "Recall", "mAP50", "mAP50-95"], [
    ["Zebra crossing", "0.968", "0.761", "0.878", "0.549"],
    ["Lane guide arrow", "0.807", "0.785", "0.861", "0.599"],
    ["**All classes**", "**0.888**", "**0.773**", "**0.869**", "**0.574**"],
  ], [3, 1.5, 1.5, 1.5, 1.5]),
  caption("Table 5. Box-level test metrics (620 frames; YOLO11n, 416 px, 12 epochs, ~2.1 h CPU training)."),
  table(["Does the frame contain a zebra crossing?", "Learned detector", "Geometric method (Milestone 2)"], [
    ["Precision", "**1.000**", "0.933"],
    ["Recall", "**0.754**", "0.235"],
    ["F1", "**0.859**", "0.376"],
    ["False alarms on 263 crossing-free frames", "**0 (0.0%)**", "6 (2.3%)"],
    ["Median latency per frame", "81 ms", "17 ms"],
  ], [4, 2.5, 2.5]),
  caption("Table 6. Head-to-head on the same 620 held-out frames, scored through the deployed ONNX model."),
  h2("5.2 Road-damage detector"),
  table(["Class", "Precision", "Recall", "mAP50", "mAP50-95"], [
    ["Longitudinal crack (D00)", ...rdClass("longitudinal_crack")],
    ["Transverse crack (D10)", ...rdClass("transverse_crack")],
    ["Alligator crack (D20)", ...rdClass("alligator_crack")],
    ["Pothole (D40)", ...rdClass("pothole")],
    ["**All classes**", ...(rdTest ? [rdTest.precision, rdTest.recall, rdTest.mAP50, rdTest.mAP50_95].map((x) => `**${f3(x)}**`) : [PENDING, PENDING, PENDING, PENDING])],
  ], [3, 1.5, 1.5, 1.5, 1.5]),
  caption(RD ? `Table 7. Road-damage test metrics on 1,166 held-out Indian images (YOLO11n, ${RD.input_size} px, ${rdEpochs} epochs, ${(RD.train_seconds / 3600).toFixed(1)} h on a laptop CPU). Serving thresholds tuned on the validation split: ${Object.entries(RD.serving_thresholds).map(([k, v]) => k.replace("_", " ") + " " + v).join(", ")}.` : "Table 7. Road-damage test metrics on 1,166 held-out Indian images. PENDING training completion; values will be copied from checkpoints/detectors/road_damage.json."),
  ...(RD ? [p(`These figures come from a short CPU training run (${rdEpochs} epochs at ${RD.input_size} px) and are a baseline. Validation mAP50 was still rising when the time budget ended (0.253 at epoch 15, 0.257 at epoch 16), so the model is under-trained; we have not measured how much longer training on a GPU improves it. Results differ sharply by class. Alligator cracks are detected best (mAP50 ${f3(rdTest.per_class.alligator_crack.mAP50)}); potholes and longitudinal cracks are fair; **transverse cracks are not detected at all** — the Indian subset gives only 50 training and 7 test boxes for that class, so its row is not a stable measurement. At image level the detector answers "does this frame show a pothole?" with precision ${f3(RD.metrics.test_image_level.per_class.pothole.precision)} and recall ${f3(RD.metrics.test_image_level.per_class.pothole.recall)}, and raises an alarm on ${(100 * RD.metrics.test_image_level.background_false_alarm_rate).toFixed(1)}% of clean-road frames. It is a screening aid for routing a survey crew, not yet a substitute for one.`)] : []),
  h2("5.3 IMU band-pass filter"),
  table(["Condition (164 held-out windows, 5 seeds)", "Without filter", "With 0.5–25 Hz filter"], [
    ["Clean recordings", "83.7%", "81.5%"],
    ["35 Hz engine vibration added", "60.1%", "**77.2%**"],
    ["Engine vibration + 12° tilted mount", "60.9%", "**78.0%**"],
  ], [4.5, 2.2, 2.3]),
  caption("Table 8. The filter costs about two points on clean logs and gains about seventeen under the vibration a bus-mounted sensor experiences. The retrained production model scores 80.5% clean and 78.7% with vibration."),
  h2("5.4 Frame-quality gate"),
  table(["Frames", "Kept / caught"], [
    ["450 real road frames (three camera sources)", "**100% kept**"],
    ["Covered lens", "100% caught"],
    ["Severe defocus", "100% caught"],
    ["Whiteout / sun glare", "98% caught"],
    ["Moderate defocus", "89% caught"],
    ["Horizontal motion blur", "4% caught (known limitation)"],
  ], [6, 3]),
  caption("Table 9. Gate calibration (scripts/calibrate_frame_gate.py)."),
  h2("5.5 INT8 quantization"),
  table(["Model", "Metric FP32 → INT8", "Latency", "Size"], [
    ["Zebra-crossing detector", "mAP50 0.864 → 0.858", "61 → 48 ms (1.3×)", "10.0 → 4.1 MB"],
    ...(QUANT.road_damage && QUANT.road_damage.int8 ? [["Road-damage detector", `mAP50 ${f3(QUANT.road_damage.fp32.mAP50)} → ${f3(QUANT.road_damage.int8.mAP50)}`, `${Math.round(QUANT.road_damage.fp32.ms_per_image)} → ${Math.round(QUANT.road_damage.int8.ms_per_image)} ms (${(QUANT.road_damage.fp32.ms_per_image / QUANT.road_damage.int8.ms_per_image).toFixed(1)}×)`, `${QUANT.road_damage.fp32.mb} → ${QUANT.road_damage.int8.mb} MB`]] : []),
    ["Classifier backbone (MobileNetV2)", "Accuracy 93.4% → 90.8%", "21 → 12 ms (1.8×)", "13.3 → 3.7 MB"],
  ], [3, 2.6, 2, 1.8]),
  caption("Table 10. INT8 results (percentile calibration). The detector loses almost nothing; the classifier loses 2.6 points, so full precision remains the default and INT8 is offered for low-power edge boards. Detector mAP50 values here are scored through the ONNX files at batch size 1, so they differ slightly from Tables 5 and 7; the FP32-to-INT8 difference is the measurement. Calibration used 128 images (48 for the road-damage detector, which ran out of memory with more). The classifier figures compare FP32 and INT8 on identical images; because the DNIT crops were regenerated, that image set is not exactly the published held-out split, so the absolute 93.4% is not comparable to the published 90.1% — the 2.6-point difference is the measurement."),
  h2("5.6 Latency"),
  p(LAT ? `On a ${LAT.frame} frame (laptop CPU, median of ${LAT.runs} runs, including the quality gate): ${Object.entries(LAT.median_ms).map(([k, v]) => k + " " + Math.round(v) + " ms").join(", ")}; classifier backbone 21 ms.`
       : "On a 1280×720 frame (laptop CPU, median of 15 runs): COCO traffic detector 202 ms, crossing detector 76 ms, both together 276 ms; classifier backbone 21 ms. The three-detector total is PENDING the road-damage model."),
];

const corrections = [
  h1("6. Corrections to the Milestone 2 Report"),
  p("Checking each Milestone 2 claim against the code and measurements produced the following corrections. We report them in the same spirit as Milestone 2's disclosure on macro-F1: a number that cannot be reproduced should not be cited."),
  table(["#", "Milestone 2 stated", "Measured / actual"], [
    ["C1", "Classifier detects 9 classes", "7 classes; the new detectors add 4 damage, 2 marking and COCO traffic classes"],
    ["C2", "10/10 subsystem tests pass", "178 automated tests, run by CI on Python 3.11 and 3.12"],
    ["C3", "IMU telemetry simulated (spring-mass-damper); 15,000 windows", "852 real field-recorded windows; 80.5% held-out"],
    ["C4", "Gatekeeper threshold: Laplacian variance 42.5", "Would drop 10.4% of real frames; calibrated gate keeps 100%"],
    ["C5", "UrbanTrafficNet 90.36% accuracy", "Not reproducible — no such trained network exists; counts now come from the COCO detector"],
    ["C6", "Faces and plates blurred on the bus", "People blurred; plates once the plate detector is trained"],
    ["C7", "Full inference cycle 76 ms", "Traffic detector 202 ms; crossing 76 ms; backbone 21 ms"],
    ["C8", "Camera height 2.45 m", "Calibration profiles use 1.45–1.52 m"],
    ["C9", "ResNet-50 is the served backbone", "A fresh installation serves MobileNetV2 (90.1%); ResNet-50 (89.2%) is an optional download"],
    ["C10", "RDD2022 India: 1,000+ frames; 47,000 photos to ingest", "7,706 labelled Indian images used; 47,420 is all six countries"],
    ["C11", "Deduplication within 8 m", "The running server used 10 m; corrected to 8 m"],
  ], [0.8, 3.7, 4.5]),
  caption("Table 11. Corrections. Evidence for each is in docs/REPORT_ALIGNMENT.md."),
];

const engineering = [
  h1("7. Software Engineering and Security"),
  bullet("**Testing.** 178 automated tests, including the web API exercised over real HTTP, per-model integration tests that verify the deployed model file against the SHA-256 in its model card, and fairness and privacy properties."),
  bullet("**Security.** The API opened any file on the server whose path was sent in an image field, and accepted a path to any file on the machine. Image fields are now decoded strictly as data, paths are confined to the dataset folder, and request sizes are capped."),
  bullet("**Dependencies.** Model files had been saved with a newer scikit-learn than the requirements allowed, so CI loaded them on a different version than the one they were trained with; the pin was corrected."),
  bullet("**Reproducibility.** Every figure in this report is regenerated by a script named in docs/REPORT_ALIGNMENT.md; detector training also runs in a provided Google Colab notebook."),
  bullet("**Licensing.** Ultralytics YOLO is AGPL-3.0 and RDD2022 is CC BY-SA 4.0 (share-alike); both affect any deployment beyond academic use and are documented in the README."),
];

const limits = [
  h1("8. Limitations and Ethical Considerations"),
  bullet("Accelerometer logs come from a road vehicle, not yet a bus; the vibration test simulates engine vibration rather than recording it."),
  bullet("Zebra-crossing data comes from three videos; performance on Indian crossings with faded paint is not yet measured."),
  bullet("The road-damage detector is under-trained (validation mAP50 still rising at the last epoch) and does not detect transverse cracks, for which the Indian subset has 50 training boxes."),
  bullet("Redaction blurs whole person boxes rather than faces, and plate redaction depends on a detector still to be trained."),
  bullet("Horizontal motion blur mostly passes the frame-quality gate."),
  bullet("No demographic inference is made about people in frame, by design; people are only counted and blurred."),
  bullet("The Priority Index removes locality from the decision, but its weights are a policy choice and should be set with municipal stakeholders."),
];

const next = [
  h1("9. Next Steps"),
  bullet(RD ? "Train the licence-plate detector and retrain the road-damage detector for 100 epochs at 640 px (GPU notebook provided)." : "Complete road-damage and licence-plate detector training (GPU notebook provided) and fill the PENDING figures."),
  bullet("Record accelerometer data on an operational bus and retrain the shock classifier."),
  bullet("Benchmark the INT8 models on a Raspberry Pi 5 and an NVIDIA Jetson."),
  bullet("Collect Indian zebra-crossing footage to measure the crossing detector on local road markings."),
  bullet("Connect live bus telemetry to the command-centre map."),
];

const refs = [
  h1("References"),
  ...[
    "[1] D. Arya, H. Maeda, S. K. Ghosh, D. Toshniwal and Y. Sekimoto, \"RDD2022: A multi-national image dataset for automatic road damage detection,\" Geoscience Data Journal, vol. 11, no. 4, pp. 846–862, 2024.",
    "[2] Z.-D. Zhang, M.-L. Tan, Z.-C. Lan, H.-C. Liu, L. Pei and W.-X. Yu, \"CDNet: A real-time and robust crosswalk detection network on Jetson nano based on YOLOv5,\" Neural Computing and Applications, vol. 34, no. 13, pp. 10719–10730, 2022.",
    "[3] G. Jocher et al., Ultralytics YOLO (YOLO11), software, AGPL-3.0, 2024–2026.",
    "[4] Microsoft, ONNX Runtime quantization documentation, 2026.",
    "[5] Augmented Startups, \"Vehicle Registration Plates Dataset,\" Roboflow Universe, CC BY 4.0, 2022.",
    "[6] J. Eriksson et al., \"The Pothole Patrol: using a mobile sensor network for road surface monitoring,\" in Proc. ACM MobiSys, 2008.",
    "[7] A. Mednis et al., \"Real time pothole detection using Android smartphones with accelerometers,\" in Proc. MobiSensor, 2011.",
    "[8] ASTM International, ASTM D6433-24: Standard Practice for Roads and Parking Lots Pavement Condition Index Surveys, 2024.",
    "[9] Ministry of Road Transport and Highways, Specifications for Road and Bridge Works (5th Revision), Section 500, Indian Roads Congress.",
    "[10] Digital Personal Data Protection Act, 2023, Government of India.",
    "[11] Smart India Hackathon 2026, Problem Statement SIH26124, Bharat Electronics Limited.",
  ].map((r) => p(r, { align: AlignmentType.LEFT })),
];

// ----------------------------------------------------------------- document
const doc = new Document({
  creator: "Team 42", title: "ROAD-SHIELD Milestone 3 Report (Draft)",
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, color: "1F3864", font: FONT }, paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, color: "2E5597", font: FONT }, paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
    alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  sections: [{
    properties: { page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
      children: [new TextRun({ text: "ROAD-SHIELD · Milestone 3 (Draft)", font: FONT, size: 16, color: "808080" })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: ["Page ", PageNumber.CURRENT, " of ", PageNumber.TOTAL_PAGES], font: FONT, size: 16, color: "808080" })] })] }) },
    children: [
      ...title,
      new Paragraph({ children: [new TextRun({ text: "Contents", font: FONT, size: 30, bold: true, color: "1F3864" })] }),
      new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
      new Paragraph({ children: [new PageBreak()] }),
      ...meta, ...abstract, ...progress, ...architecture, ...data, ...methods, ...results,
      ...corrections, ...engineering, ...limits, ...next, ...refs,
    ],
  }],
});

const out = path.resolve(__dirname, "..", "..", "docs", "CSET485_ROAD_SHIELD_Milestone3_Report_Draft.docx");
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(out, buf); console.log("wrote", out, buf.length, "bytes"); });
