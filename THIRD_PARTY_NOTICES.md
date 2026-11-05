# Third-party notices

OmniMemBench is released under the [Apache License 2.0](LICENSE). The components below are adapted from third-party projects and remain under their original licenses.

## Mem-Gallery

- Source: <https://github.com/YuanchenBei/Mem-Gallery> (`benchmark/memengine/`)
- License: MIT, Copyright (c) 2026 Yuanchen Bei
- Used in: the memory framework and evaluation script under `omnimembench/engine/` (`memory/`, `operation/`, `function/`, `utils/`, `config/Config.py`, and `evaluate/evaluation.py`), modified and extended for OmniMemBench. Mem-Gallery's framework is based on MemEngine (below).

```
MIT License

Copyright (c) 2026 Yuanchen Bei

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## MemEngine

- Source: <https://github.com/nuster1128/MemEngine>
- Authors: Zeyu Zhang, Quanyu Dai, Xu Chen, Rui Li, Zhongyang Li, Zhenhua Dong
- License: MIT, as declared in the MemEngine README. The upstream repository does not include a license file or copyright line.
- Used in: the parts of `omnimembench/engine/` that Mem-Gallery inherits from MemEngine. `function/Judge.py`, `function/Reflector.py`, `function/Retrieval.py`, `function/Summarizer.py`, `utils/AutoSelector.py`, and `utils/Client.py` are unchanged from MemEngine.

```
MIT License

Copyright (c) MemEngine authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## HippoRAG

- Source: <https://github.com/OSU-NLP-Group/HippoRAG> (`src/hipporag/`)
- License: MIT, Copyright (c) 2025 OSU Natural Language Processing
- Used in: `omnimembench/engine/baselines/hipporag/`, modified for OmniMemBench. The full license text is in [`omnimembench/engine/baselines/hipporag/LICENSE`](omnimembench/engine/baselines/hipporag/LICENSE).

## A-MEM

- Source: <https://github.com/agiresearch/A-mem> (`agentic_memory/`)
- License: MIT, Copyright (c) 2025 AGI Research
- Used in: `omnimembench/engine/baselines/amem/`, modified for OmniMemBench. The full license text is in [`omnimembench/engine/baselines/amem/LICENSE`](omnimembench/engine/baselines/amem/LICENSE).
