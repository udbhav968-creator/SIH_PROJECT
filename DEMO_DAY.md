# ROAD-SHIELD — demo day sheet

SIH 2026 · Problem statement SIH26124 · Bharat Electronics Limited

---

## 1. Before you leave (10 minutes)

```powershell
cd $env:USERPROFILE\Desktop\SIH_PROJECT
$py = ".\.venv\Scripts\python.exe"

& $py -m scripts.fix_label_conflicts                                  # once
& $py -m scripts.fetch_cracks_potholes_dataset --limit 2235 --workers 12
& $py -m training.train_mega_suite                                    # ~3 min
& $py -m api.server
```

Open `http://127.0.0.1:8000/dashboard`. Check three things:

1. Header says **engine online · 2 models loaded**
2. Model card tab shows **85.8%**, not 36.6%
3. Upload any road photo — a box is drawn and numbers appear

If the object detector is installed (`checkpoints/road_shield_detector.onnx`),
the health endpoint also lists `coco_object_detector`, and photos with people
or vehicles get a second set of boxes.

**Have a fallback.** Screenshot each tab now and put the images in your slides.
If the venue wifi or the laptop misbehaves, you still have the evidence.

---

## 2. The three-minute demo

**Open on the Inspection tab, photo already chosen.**

> "ROAD-SHIELD turns an ordinary road photograph into a costed repair order.
> Everything you're about to see is computed live on this laptop."

**Drop the photo in.** While it runs:

> "The image goes to a trained classifier, a geometry stage that converts
> pixels to square metres, and a costing stage that uses the MoRTH Section 500
> bitumen tables."

**Point at the result:**

> "Pothole, 48% confidence, 6.6 square metres, 2cm deep, 0.36 tonnes of mix,
> ₹2,737 to repair. The confidence is honest — this is a hard photograph."

**Switch to Works & costing. Click Generate work order, then Tamper.**

> "Every work order is sealed with SHA-256. I'll change the budget field and
> re-verify — the seal breaks. A contractor cannot quietly inflate a bill."

**Switch to Corridor map.**

> "Five buses reported four defects here. Two reports were the same pothole,
> five metres apart, so the system merged them by GPS distance. You pay to fix
> it once."

**Finish on Model card.**

> "And here is what we actually measured: 85.8% on 450 photographs the model
> never saw. Cracks 0.91 F1, potholes 0.75. Damaged signs score zero, because
> we only have three examples of them. We'd rather show you that than hide it."

---

## 3. Questions judges will ask

**"What's your accuracy?"**
85.8% on a held-out set of 450 photographs, macro-F1 0.67. Cracks 0.91,
potholes 0.75. Four classes have fewer than 20 examples and perform poorly —
that's a data volume problem and it's visible on the model card.

**"How do I know you're not testing on training data?"**
We split at the level of the source photograph, not the file. Augmented copies
of a photograph stay with their original, so no version of a test image is
ever seen in training. We also hash every image and check for near-duplicates
across the split — `scripts/validate_models.py` runs that audit.

**"Where did the data come from?"**
2,235 photographs from DNIT, the Brazilian federal highway department, with
1,921 crack and 564 pothole annotations. We crop each annotated defect into a
training example. Four smaller classes come from Wikimedia Commons. We are not
claiming Indian road data yet — that's RDD2022, which is our next step.

**"Is this deep learning?"**
The classifier today is a support vector machine on engineered features — HOG,
local binary patterns and colour histograms. The fine-tuning script for a
pretrained CNN is written and in the repo; PyTorch wouldn't load on this laptop
this morning. Object detection uses a YOLOv8 network trained on COCO's 330,000
images, running through ONNX Runtime.

**"Why is one class at zero?"**
Damaged traffic signs: three examples in the test set, fourteen in total. No
model learns a class from fourteen pictures. We report it rather than dropping
the class to flatter the average.

**"Could a contractor game this?"**
Three defences. Work orders are SHA-256 sealed, so fields can't be edited after
issue. Before-and-after repair photos are compared with SSIM and Laplacian
variance, so a re-submitted old photo is rejected. And GPS deduplication means
the same pothole can't be billed twice from two reports.

**"What's the inference time?"**
Measured, not estimated: 19ms for feature extraction, 159ms for the full
pipeline per image at the median, on this laptop CPU. About 6 images a second.

**"What would you do with more time?"**
RDD2022's 47,000 annotated Indian road images, fine-tune the CNN on a GPU, and
replace the simulated IMU data with real accelerometer recordings from a
vehicle. All three are prepared in the repo and waiting on compute.

---

## 4. Things not to claim

- Not trained on Indian roads yet. The photographs are Brazilian.
- The IMU model's 100% accuracy is on simulated data, and means little.
- There is no dashcam video in the project; it analyses photographs.
- The system does not identify people. It detects that a person is present,
  which is what pedestrian safety needs, and nothing more.

Saying these before a judge finds them is worth more than the marks you'd lose
by having them found.

---

## 5. The story that wins

Most teams show a number. Your story is stronger:

> "We started at 36.6%. We audited our own dataset and found twenty
> photographs filed under three contradictory labels at once — the same
> picture labelled normal, cracked and potholed. We fixed it: 79.4%. Then we
> brought in 1,700 real annotated defects from a national highway department:
> 85.8%. Every number on that screen is measured, and the code that measures it
> is in the repository."

That is an engineering team talking, not a demo.
