# Current Usage
- View the form of data with `example_usage.py`. Other files are only used to export the data we need.
- install dependencies with `pip install -r requirements.txt`
- initialize submodule with `git submodule update --init --recursive`
- `cd vector-quantize-pytorch`, and install the submodule with `pip install -e .`

# Change Log

- [24.12.02] 导出脚本汇总于`exporter`

- [24.12.05] 新增`dataloading.py`
    - `ChunkedDataset`可用于训练VQ-VAE读取数据

- [24.12.05] 新增vector-quantize-pytorch作为submodule
    - 后续开发可以继承submodule中的类，避免复制粘贴一堆代码
    - 新增VectorQuantize的例子，后续会继续添加Validation和Test以及更多模型的支持