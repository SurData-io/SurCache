# SurCache

SurCache 是 SurData 里、`surdata-sdk` 以外的 KV Cache 层。推理引擎仍是 vLLM。缓存引擎有两条：

- **LMCache 路径**已经单独开源：[SurData-io/lmcache](https://github.com/SurData-io/lmcache)。那是 LMCache 0.5.3 的 Apache-2.0 分叉，加上页直写的 GDS 路径。描述里写明 SDK 保持闭源。
- **直连路径**在本仓库的 `python/surdata_vllm` 和 `native/`。vLLM 0.33 的 `KVConnectorBase_V1` 实现是 `SurDataKVConnector`。调度只投递工作，主机与 GPU 之间的搬运在连接器自己的线程里做。这条路径不 `import lmcache`。

论文在 [PAPER.md](PAPER.md)。它只写一件事：搬运拆成七条主机线程和三条 CUDA 流，彼此不等。拷贝流只等生产这块数据的 event，计算流不回头等拷贝。

## 不在本仓库里的部分

- `surdata-sdk`：前缀哈希、命名空间、异步读写、durable commit。闭源。
- SurFile：SurData File System。内核数据面模块按产品说明是专有的，GPL 前端只做 VFS。元数据节点和存储节点也不在这里。

直连连接器运行时要链上 SDK，并有一套已经起来的 SurData 存储。本仓库单独编过传输库的自测，不能单独完成一次推理。

## 许可

本仓库是 [Apache License 2.0](LICENSE)，与 vLLM、LMCache 相同。修改过的文件要标明改动，并保留原有声明。`NOTICE` 写了这两份上游。

Python 导入名仍是 `surdata_vllm`。论文和仓库名用 SurCache，避免和闭源 SDK 混成一个名字。

## 固定的直连内存

直连 DRAM 的两个池大小是固定的，合计 20GiB，不按机器内存改：

- store 池：`8589934592`（8GiB），承接 GPU 到主机的保存
- preload 池：`12884901888`（12GiB），承接命中后、拷回 GPU 之前的读入

## 已完成

- 论文：七条主机线程、计算流 / D2H 流 / H2D 流，以及谁不等谁。
- 直连连接器源码：Python 包、单测、`libsurdata_transfer` 的 C++ 源。
- Apache-2.0 许可和 `NOTICE`。

## 尚未完成

- 生产保存走 torch 的 store 流。传输库里的 D2H 流还在，生产路径不提交给它。
- `per_layer` 仍让 D2H 流等待整条计算流。只等本步 event 的是 `end`。
- 本仓库不包含可运行的 SDK，也不能代替 SurFile。
- 导入名还没有改成 `surcache`。改名会打断现有的 `kv_connector_module_path`。
