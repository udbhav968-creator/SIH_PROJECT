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
& $py -m scripts.fetch_cnn_backbone                                   # 98 MB, once
& $py -m training.train_cnn_head --compare                            # ~2 min
& $py -m api.server
```

Open `http://127.0.0.1:8000/dashboard`. Check four things:

1. The server printed `✓ Vision classifier: deep CNN embeddings
   (cnn:resnet50+logistic)` — if it says `HOG/LBP + SVM baseline` instead, the
   backbone download didn't finish and you're demoing the 82.6% path, not 89.2%
2. Header says **engine online · 2 models loaded**
3. Model card tab shows **89.2%**, not 36.6%
4. Upload any road photo — a box is drawn and numbers appear

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

> "And here is what we actually measured: 89.2% on images from 356 photographs
> the model never saw, split by source photograph so no augmented copy leaks
> across. Normal road 0.94 F1, potholes 0.90, cracks 0.87. The four rare classes
> have two or three test images each, so their scores aren't stable — we'd
> rather show you that than average it away."

---

## 3. Questions judges will ask

**"What's your accuracy?"**
89.2%, macro-F1 0.675, on 390 images from 356 photographs the model never saw.
Normal road 0.94 F1, potholes 0.90, cracks 0.87 — those three carry 380 of the
390 test images. The other four classes have two or three test images each and
their scores swing on a single image, which is what drags macro-F1 down. Both
numbers are on the model card; we didn't pick the flattering one.

**"Why is macro-F1 so much lower than accuracy?"**
Because four of the seven classes have almost no data. That gap is the honest
statement of what this system still needs, and no change of architecture closes
it — only more photographs of waterlogging, zebra crossings and dividers.

**"What model is it?"**
A ResNet-50 trained on ImageNet, frozen, used as a feature extractor through
ONNX Runtime; a class-balanced logistic head learns the mapping from its
embeddings to the seven road classes. With roughly 1,600 images, training only
the head beats fine-tuning the whole network and takes two minutes on a CPU.
The comparison against hand-engineered features on the identical split —
89.2% versus 82.6% on the identical split — is in
`checkpoints/cnn_head_resnet50_report.json`, produced by
`python -m training.train_cnn_head --compare`.

**"Why not fine-tune, or use a transformer?"**
Both were available; neither is justified by 1,600 images. Fine-tuning ResNet-50
end to end on this dataset overfits, and we can't demonstrate it on this laptop
because PyTorch won't load on it. What we can demonstrate, we measured.

**"How do I know you're not testing on training data?"**
We split at the level of the source photograph, not the file. Augmented copies
of a photograph stay with their original, so no version of a test image is
ever seen in training. We also hash every image and check for near-duplicates
across the split — `scripts/validate_models.py` runs that audit.

**"Where did the data come from?"**
2,373 distinct photographs. The core is 2,235 from DNIT, the Brazilian federal
highway department, with 1,921 crack and 564 pothole annotations, each cropped
into a training example. On top of that, four Kaggle datasets pulled through the
official API by `scripts/fetch_kaggle_datasets.py`. Every incoming image is
perceptually hashed against what we already hold — one of those Kaggle sets
turned out to be 94% a re-upload of another, and 351 of its 374 images were
rejected as duplicates. Without that check they would have landed in training
and test both. We are not claiming Indian road data yet; that's RDD2022, next.

**"Is this deep learning?"**
Yes. A ResNet-50 trained on ImageNet's 1.28 million images does the seeing; a
small classifier learns the mapping from its embeddings to our seven classes.
The network is frozen rather than fine-tuned, which is the correct choice at
2,400 images. It runs through ONNX Runtime, not PyTorch — PyTorch will not load
on this laptop, and not depending on it turned out to be an advantage. Object
detection is YOLOv8 trained on COCO's 330,000 images, also through ONNX Runtime.
The hand-engineered HOG/LBP pipeline is still in the repo as the fallback, and
we score it on the identical split every run: 82.6% against the CNN's 89.2%.

**"Why are some classes so weak?"**
Waterlogging, zebra crossings, dividers and damaged signs have two or three test
images each. No model learns a class from a handful of pictures. We report them
rather than dropping them to flatter the average.

**"Could a contractor game this?"**
Three defences. Work orders are SHA-256 sealed, so fields can't be edited after
issue. Before-and-after repair photos are compared with SSIM and Laplacian
variance, so a re-submitted old photo is rejected. And GPS deduplication means
the same pothole can't be billed twice from two reports.

**"What's the inference time?"**
Measured, not estimated: about 80 ms for the ResNet-50 embedding on this laptop
CPU, inside a full pipeline that also does geometry, costing and PCI. No GPU
anywhere. On a slower machine we can swap in MobileNetV2 — 14 MB instead of
98 MB, roughly three times faster, a few points less accurate — and the loader
pairs each classifier head with the backbone it was trained on so the two can
never be mixed.

**"What would you do with more time?"**
Photographs of the four starved classes — that is the binding constraint, not
the model. Then RDD2022's 47,000 annotated Indian road images, fine-tuning on a
GPU, and real accelerometer recordings to replace the simulated IMU data. All
three are prepared in the repo and waiting on compute.

---

## 4. Things not to claim

- Not trained on Indian roads yet. The photographs are Brazilian.
- The IMU model's 100% accuracy is on simulated data, and means little.
- There is no dashcam video in the project; it analyses photographs.
- The system does not identify people. It detects that a person is present,
  which is what pedestrian safety needs, and nothing more.
- The damaged-sign class has 2 test images. During development it briefly
  scored 0.98 F1 because a Kaggle dataset of *road signs* had been filed under
  *damaged road signs* - the model had learned "a sign is here", not "this sign
  is broken". Removing those 299 images raised overall accuracy from 88.5% to
  89.2%. If a judge asks about the weakest part of the system, this is a better
  answer than a defensive one.

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
