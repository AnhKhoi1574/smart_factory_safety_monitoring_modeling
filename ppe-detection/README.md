# PPE detection

The active workstream is [PPE v2](v2_pipeline/README.md). It trains an object
detector for `person`, `helmet`, `vest` and `cleaning_coverall`.

The detector returns objects. Tracking, person-to-PPE association and compliance
rules belong to the downstream application.

| Area | Use |
|---|---|
| [v2 pipeline](v2_pipeline/README.md) | Current data preparation and training workflow |
| [v1 legacy](v1_legacy/README.md) | Historical notebooks and preprocessing notes |
| [Model summary](../docs/MODELS.md) | Recorded results and delivered model files |
| [Handover](../docs/HANDOVER.md) | Receive the repository, datasets and models |
