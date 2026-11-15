# Supplied sample data

These four images and the 66 question rows were supplied with this project request. The rows cover 33 visual facts in English and Finnish. They are included for local smoke tests and evaluation; their original rights and dataset conditions remain applicable. No rights to the underlying drawings are granted by this code.

Do not use these four plans to claim general accuracy. Split larger datasets by source property/plan, preserving near-duplicate images and bilingual question pairs together.

Run from the repository root:

```bash
drawingqa scan --data-root examples/data
drawingqa evaluate --config configs/models.yaml --questions examples/questions_sample.jsonl --data-root examples/data --models qwen nemotron --output runs/sample
```

The evaluation command requires running model endpoints and can incur inference charges when configured for a hosted provider. Expected answers never enter model prompts.
