---
title: ROAD-SHIELD Engine
emoji: 🛣️
colorFrom: gray
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
short_description: Road-defect AI engine behind the ROAD-SHIELD website
---

# ROAD-SHIELD live engine

The inference engine behind the ROAD-SHIELD website (SIH 2026, SIH26124). A road photograph goes in, and out
come a classification, a defect mask, the area, a depth interval and a cost range.

- Health: `/api/v1/health` · readiness: `/api/v1/ready` · API reference: `/api-docs`
- Public demo: uploads are analysed and not stored, write endpoints are locked, and each visitor gets
  20 model requests per minute.
- Source code, models and the measurement reports are at https://github.com/udbhav968-creator/SIH_PROJECT
