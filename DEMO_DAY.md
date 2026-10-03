# ROAD-SHIELD — demo day sheet

SIH 2026 · Problem statement SIH26124 · Bharat Electronics Limited

Every number on this sheet comes from a file in `checkpoints/`. If the Colab
run finished and served a fine-tuned network, the site shows *that* number —
**read the number off the Models page and say that one**, not the one printed here.

---

## 1. Before you leave (10 minutes)

```powershell
cd C:\Users\Dell\SIH_PROJECT
git pull
python -m unittest discover -s tests -t .        # should end in OK
python -m api.server                             # leave this window open
```

Open `http://127.0.0.1:8000/` and check:

1. The server printed which vision classifier it loaded (fine-tuned CNN or
   `cnn:mobilenetv2` head). If it says `HOG/LBP + SVM baseline`, the backbone is
   missing — run `python -m scripts.fetch_cnn_backbone --model mobilenetv2`.
2. Overview: four headline cards show numbers, not dashes.
3. Models page: held-out accuracy and the per-class table load.
4. Inspection → "Use a dataset photograph" draws a mask and shows a cost range.
5. Architecture page: the claims table and the corrections table load.

**Have a fallback.** Screenshot every page now and put them in the slides. The
Corridor map needs internet for tiles; without it the page draws a plain plot of
the same points, so it still works — but screenshots are safer.

---

## 2. The three-minute demo

**Open on Inspection, photograph already chosen.**

> "ROAD-SHIELD turns a photograph from a bus camera into a costed, sealed repair
> order. Everything you see is computed live on this laptop, on the CPU."

**Click Analyse.** While it runs:

> "A classifier decides what the defect is, a segmenter draws its outline, the
> camera geometry turns pixels into square metres, and the MoRTH Section 500
> tables turn that into tonnes of mix and rupees."

**Point at the result** — read what is on screen, and point at the badges:

> "Green numbers are measured. Amber ones are estimates and carry a range —
> depth, for example, is never a measurement from one photograph, so we show an
> interval rather than a fake precise figure."

**Click "Blur people & plates".**

> "Before any image leaves the system, people and number plates are blurred, for
> the DPDP Act. We haven't measured its recall yet, and the screen says so."

**Switch to Works. Generate a work order, then Tamper.**

> "Every work order is sealed with SHA-256. I change the budget and re-verify —
> the seal breaks. And if a report arrives without GPS, the order is held, not
> given an invented location."

**Switch to Corridor.**

> "Several buses report the same pothole. Reports of the same defect class
> within 8 metres merge into one, so you pay to fix it once. The raw sightings
> are kept as evidence that the merge happened."

**Finish on Models.**

> "Here is what we measured: 87% on 625 images from 589 photographs the model
> never saw, split by source photograph so no copy leaks across. On Indian
> roads — the official RDD2022 India test split, never used in training — 80%.
> Before we added Indian data it was 33%. The four rare classes have seven or
> eight test images each, so their scores swing — we show that rather than
> average it away."

---

## 3. Questions judges will ask

**"What's your accuracy?"**
87.0%, macro-F1 0.70, on 625 held-out images from 589 photographs. On Indian
roads, 80.4%, macro-F1 0.78, on 997 crops from the RDD2022 India test split.
Normal road, crack and pothole are each around 0.87–0.89 F1 and carry 595 of the
625 test images. (If the fine-tuned CNN is served, quote the Models page.)

**"Why is macro-F1 lower than accuracy?"**
Four of the seven classes — waterlogging, zebra crossings, dividers, damaged
signs — have about 50 photographs each and 7–8 test images. One image moves
their F1 by more than 0.1. Only more photographs fixes that.

**"What model is it? Is this deep learning?"**
Yes. A MobileNetV2 CNN pretrained on ImageNet extracts features, with a flipped
copy averaged in; an SVM head maps them to our seven classes. It runs on ONNX
Runtime, no GPU. We also fine-tuned EfficientNet-B0/B2, MobileNetV3 and
ResNet-50 end to end on a Colab GPU. Which one is served is decided by a rule
fixed *before* looking at test scores: the fine-tuned network wins only if it
beats the head on validation accuracy *and* macro-F1. Object detection is
YOLOv8n trained on COCO. Hand-crafted HOG/LBP features score 79.0% on the same
split — that's the baseline we beat.

**"How do I know you're not testing on training data?"**
The split is by source photograph, not by file, so augmented copies stay with
their original. Every image is perceptually hashed and near-duplicates are
rejected — one Kaggle set was 94% a re-upload of another and 700 of its 739
images were dropped. The Indian test crops come from RDD2022's own test split
and are never written to a training folder.

**"Where did the data come from?"**
DNIT, the Brazilian highway department (2,235 photographs, 4,720 hand-drawn
polygons — the segmenter trains on those); RDD2022 India, 2,173 training
photographs; three Kaggle pothole sets; and 207 Wikimedia and field photographs
for the rare classes. We *excluded* 622 concrete-wall crack close-ups: correctly
labelled, but not road scenes, and they were teaching the model that a linear
feature on grey is "clean road".

**"Is the IMU data real?"**
Yes — 10 drive logs on Indian roads, 205,491 samples at 100 Hz, from a car.
RandomForest scores 87.2% on 164 held-out windows split by time block. A 1-D CNN
is compared by 5-fold CV and served only if it is better. Not yet from a bus.

**"Could a contractor game this?"**
Three defences: SHA-256 sealed work orders; before/after repair photos compared
with SSIM, sharpness and perceptual hashes so an old photo is rejected; and
class-aware GPS deduplication so one pothole can't be billed twice.

**"How accurate is the area and the cost?"**
Area is measured *if* the camera is calibrated: on one photograph, the assumed
mount gave 0.007 m² and the calibrated one 0.049 m² — seven times. That's why
every vehicle gets a calibration profile and every number carries its
provenance. Depth is an estimate with an interval, so cost is a range.

**"What's the inference time?"**
About 30 ms for the two MobileNetV2 embeddings; the full pipeline is 1.7–2.4 s
on a single-core VM, dominated by the pixel segmenter.

**"What would you do with more time?"**
A bus-mounted pilot: real bus camera footage and accelerometer logs, measured
pothole depths for ground truth, more photographs of the four rare classes, and
an annotated set to measure the privacy blur's recall.

---

## 4. Things not to claim

- Indian photographs are RDD2022 smartphone images; none were taken from a bus.
- The Indian test covers three classes (normal, crack, pothole), not seven.
- IMU logs are from a car, not a bus fleet.
- Depth is an estimate, never a measurement.
- PCI and deterioration models reproduce engineering formulas (ASTM D6433,
  HDM-4 style); they are not validated against field surveys.
- Privacy blur recall has not been measured.
- The system does not identify people; it detects that a person is present.

Saying these before a judge finds them is worth more than the marks you'd lose
by having them found. The Architecture page lists every claim we withdrew.

---

## 5. The story that wins

> "We started at 36.6%. We audited our own dataset and found twenty photographs
> filed under three contradictory labels at once. Fixing that took us to 79%.
> Real annotated defects from a national highway department and ImageNet
> features took us to 89% — but on Indian roads that model scored 33%. So we
> brought in RDD2022 India, threw out data that wasn't road scenes, and now
> score 80% on an Indian test set the model never saw. Every number on the
> screen is measured, and the code that measures it is in the repository."

That is an engineering team talking, not a demo.
