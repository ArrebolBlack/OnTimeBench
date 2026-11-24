# OnTimeBench licensing

Apache 2.0 covers the original benchmark software, including the evaluation code,
execution settings and MJCF task definitions ([license text](LICENSE)).

## Training demonstrations

The OnTimeBench authors release all 11,200 training demonstrations, their
annotations and the training manifest under CC BY 4.0. See the
[dataset card](https://huggingface.co/datasets/JiaqiYin/OnTimeBench) and the
[full data license](LICENSE-DATA.txt).

## Evaluation data

The 1,400 test scenes in [`manifests/v1/test_id.jsonl`](manifests/v1/test_id.jsonl)
and the per-scene latency streams in
[`manifests/v1/latency_streams.json`](manifests/v1/latency_streams.json) are
released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode.en).

When sharing these data or an adaptation, credit the OnTimeBench authors, link
to the source and license, and indicate any changes. A suggested credit is
“OnTimeBench evaluation data, by the OnTimeBench authors, licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).”

## Third-party assets

The surface texture `assets/textures/red-wood.png` comes from
[robosuite 1.4.0](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/robosuite/models/assets/textures/red-wood.png)
and keeps its upstream [license](assets/textures/robosuite/LICENSE) and
[author list](assets/textures/robosuite/AUTHORS).
