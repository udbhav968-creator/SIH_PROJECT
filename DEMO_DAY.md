# ROAD-SHIELD — demo day sheet

SIH 2026 · Problem statement SIH26124 · Bharat Electronics Limited

Every number here comes from a file in `checkpoints/`, and the site reads the same
files. If a number on screen ever differs from this sheet, **say the one on screen**.

---

## 1. Before you leave (10 minutes)

```powershell
cd C:\Users\Dell\SIH_PROJECT
git pull
python -c "import sklearn; print(sklearn.__version__)"   # must print 1.8.x
python -m api.server                                     # leave this window open
```

If scikit-learn is not 1.8.x: `pip install "scikit-learn>=1.8,<1.9"`, then start the
server again. With the wrong version the segmenter cannot load: the page still works,
but no defect area or cost can be measured.

Open `http://127.0.0.1:8000/` and check:

1. The server printed `✓ Vision classifier: fine-tuned CNN (…mobilenet_v3_large_refit.onnx)`
   **and** `✓ Defect segmenter: pixel classifier masks`. (The pixel classifier serves: it
   beat the U-Net in the end-to-end check, see `checkpoints/segmenter_selection.json`.)
2. Overview shows **93.3%** and **92.3%** (Indian roads) — not dashes.
3. Inspection → "Use a dataset photograph" draws a mask and gives an area and a cost range.
4. Models, Data, Architecture and Design pages load.

**Have a fallback.** Screenshot every page and put them in the slides. The live site
(road-shield-ai-engine.vercel.app) shows every result, but photo analysis needs the laptop.

---

## 2. The three-minute pitch

**0:00 — The problem.**
> "Potholes caused 4,446 road accidents and 1,856 deaths in India in 2022, by the
> Ministry of Road Transport's own count. Roads are inspected by people, occasionally,
> and repairs are billed with no independent record. ROAD-SHIELD turns the city's buses,
> which already drive every road every day, into the inspection network."

**0:30 — Live demo: Inspection.** Pick a photograph, click Analyse.
> "A fine-tuned CNN classifies the road, a segmenter outlines the defect, the bus's own
> camera geometry turns pixels into square metres, and the MoRTH Section 500 tables turn
> that into tonnes of bitumen mix and rupees. Green numbers are measured; purple ones are
> estimates and carry a range — depth from one photograph is never a measurement, so we
> show an interval, not a fake precise figure."

Click **Blur people & plates**:
> "Before any image leaves the system, faces and number plates are blurred."

**1:15 — Works.** Generate a work order, then Tamper.
> "Every order is sealed with SHA-256. Change one rupee and the seal breaks. No location,
> no order: it is held, never given an invented address."

**1:35 — Corridor.**
> "Five buses report the same pothole; reports of the same class within 8 metres merge
> into one defect, so it is fixed — and paid for — once."

**1:55 — Models.**
> "93.3% on 630 photographs the model never saw. 92.3% on held-out Indian road
> photographs — and before we added Indian data, the same pipeline scored 33% there. We
> compared four deep networks and picked the winner on validation data only; the test set
> was scored once. It runs in 7 milliseconds per image on an ordinary CPU, so it can run
> on the bus itself."

**2:30 — Honesty and scale.** Architecture page, then Design page.
> "Every claim has its evidence file, and these are the claims we withdrew when we audited
> ourselves. The design scales to a 5,000-bus fleet on about forty CPU cores, because the
> bus discards 95% of frames before uploading anything. The Impact page shows the cost per
> kilometre surveyed and how we meet each of Microsoft's six Responsible AI principles —
> including where we don't yet. What we need next is one bus route for a 90-day pilot."

---

## 3. Questions judges will ask

**"What's your accuracy?"**
93.3%, macro-F1 0.828, on 630 held-out images from 567 photographs. On Indian roads,
92.3% (macro-F1 0.922) on 1,200 crops from 781 held-out RDD2022 India photographs.
Normal road, crack and pothole are each 0.93–0.97 F1.

**"Is it deep learning? Is it just a pretrained model?"**
Deep learning, with transfer learning. The networks start from ImageNet weights, then
**every layer** is trained on our road photographs on a GPU. We fine-tuned four:
EfficientNet-B0 (91.3% test), EfficientNet-B2 (91.4%), MobileNetV3-Large (91.4%) and
ResNet-50 (89.5%). MobileNetV3-Large had the best validation score, was retrained on
train+validation, and scores 93.3%. It replaced a frozen-feature model (87.9%) only
because it beat it on validation accuracy *and* macro-F1. The baseline with hand-crafted
features scores 84.9% on the same split.

**"Why is macro-F1 lower than accuracy?"**
Four classes — waterlogging, zebra crossings, dividers, damaged signs — have about 50
photographs each and 7–8 test images. One image moves their F1 by more than 0.1. More
photographs fix that; a bigger model does not.

**"How do I know you're not testing on training data?"**
Splits are by source photograph, never by file. Every image is perceptually hashed and
near-duplicates rejected (one Kaggle set was 94% a re-upload; 700 of 739 dropped). The
Indian test photographs are held out by photograph and never written to a training
folder. Model choice uses validation data only.

**"Why not a deep segmenter?"** (the best answer you have — use it)
We trained one: a U-Net with a ResNet-18 encoder. On its own dataset's 500 test photos it
outlines potholes four times better than the pixel classifier (IoU 0.641 vs 0.144) and
falsely marks 1.7% of clean roads instead of 23.3%. We served it — then a test failed, so
we ran both through the whole pipeline on photographs from other datasets. The U-Net found
9 of 24 defects with 6 false alarms out of 36 clean roads; the pixel classifier found 24 of
24 with 3. The rule kept the pixel classifier. A better mask on its training data was not a
better product; the U-Net needs more varied training photos, and that is next.

**"Did anything go wrong?"** (a strong answer — use it)
Yes, and we fixed it in the open. Our first IMU comparison shuffled 1-second windows into
cross-validation folds, so neighbouring windows leaked across folds and it picked a CNN.
The time-separated held-out logs disagreed (CNN 78% vs RandomForest 87.2%). We switched to
time-blocked folds; the fair comparison chose the RandomForest. It is on the corrections
list.

**"Is the IMU data real?"**
Yes — 10 drive logs on Indian roads, 205,491 samples at 100 Hz, from a car. RandomForest
87.2% on 164 held-out windows split by time block. Not yet from a bus.

**"Could a contractor game this?"**
SHA-256-sealed work orders; repair photos compared with SSIM, sharpness and perceptual
hashes so an old photo is rejected; class-aware GPS deduplication so one pothole cannot be
billed twice.

**"How accurate are area and cost?"**
Area is measured *if* the camera is calibrated: on one photograph the assumed mount gave
0.007 m² and the calibrated one 0.049 m² — seven times. Each vehicle gets a calibration
profile. Depth is an interval, so cost is a range. The outline comes from the pixel
classifier: crack IoU 0.231 and pothole IoU 0.144 on 500 held-out photographs. Outlines
are the weakest link, and stated as such - which is why we trained a U-Net (next answer).

**"How fast is it?"**
7.2 ms per image for the classifier on a CPU. The full pipeline (classify, segment,
geometry, cost, seal) takes about 2.6 s per frame on a 2-core cloud machine, mostly the
pixel segmenter.

**"How does it scale?"**
The bus classifies frames itself and uploads only the ~5% that are not normal road; the
server skips repeat sightings of known defects. For 5,000 buses that is ~105 uploads per
second and ~20–40 CPU cores. The Design page has the calculator; the numbers are planning
estimates, not measurements.

**"What would you do next?"**
A one-route bus pilot (real footage and IMU from a bus), measured pothole depths for
ground truth, more photographs of the rare classes, and an annotated set to measure the
privacy blur.

---

## 4. Things not to claim

- Indian photographs are RDD2022 smartphone images; none from a bus.
- The Indian test covers three classes (normal, crack, pothole), not seven.
- IMU logs are from a car, not a bus fleet.
- Depth is an estimate, never a measurement.
- PCI and deterioration reproduce engineering formulas; not validated against field surveys.
- Privacy blur recall has not been measured.
- Capacity numbers (5,000 buses, 40 cores) are planning estimates.

---

## 5. The story that wins

> "We started at 36.6%. Our own audit found twenty photographs filed under three
> contradictory labels — fixing that took us to 79%. Real annotated highway data and
> ImageNet features took us to 89%, but on Indian roads that model scored 33%. So we
> brought in RDD2022 India, threw out data that wasn't road scenes, fine-tuned four deep
> networks and let validation data pick the winner: 93.3% overall and 92.3% on Indian
> roads it never saw, at 7 milliseconds an image. Every number on screen is measured, and
> every claim we got wrong is listed next to the ones we got right."
