# SurCache：把 KV 的搬运拆成互不等待的流

SurData  
2026-11-13

## 摘要

vLLM 的计算跑在一条 CUDA 流上。若保存 KV Cache 也排在这条流上，或者某条主机线程在算下一步之前先等落盘结束，则拷贝和存储的延迟会串进下一次前向。SurCache 是 SDK 以外的这一层。它把搬运拆开：调度只发布相位；计算线程只录 event、把块交给队列；七条主机线程各等自己的唤醒，不调用彼此的收尾函数；GPU 上计算流、D2H 流、H2D 流分开。拷贝流只等「生产这块字节」的 event。计算流不等拷贝流。H2D 不等 D2H。查找不等提交，提交不等查找。

## 1. 一条流在等什么

CUDA 流是 GPU 上的命令队列。同一条流里的内核和 `cudaMemcpyAsync` 按提交顺序执行。不同的流可以同时推进。`cudaStreamSynchronize` 或在计算流上等待拷贝流，会让 CPU 或下一次前向停到拷贝结束。

主机线程是另一层。线程在 `join`、在锁里调用对方的慢函数、或在持有步进锁时等待磁盘，都会把对方的延迟变成自己的延迟。SurCache 要拆开的是这两层，不是只多开几个线程名字。

KV Cache 是每一层、每个已生成 token 的键和值。vLLM 把它切成块，放在 GPU 高带宽内存里。块要离开 GPU 时经历四段：计算写完这块，D2H 拷到钉住的主机内存，SDK 把这块写入 SurFile，commit 把前缀发布到索引。命中后反向再来三段：查找、读进 preload 池、H2D 拷回 GPU。这七段若接成一条调用链，最慢的一段决定下一步能不能开始。

## 2. 计算线程不搬运

vLLM 0.33 里，直连连接器是 `KVConnectorBase_V1` 的 `SurDataKVConnector`。调度角色和 worker 角色各有一份。调度侧只有状态板和 `SchedulerFront`，没有传输线程。worker 侧才启动搬运。

计算线程在两处碰到缓存。`save_kv_layer` 在每一层前向之后录一个 CUDA event，或者在 `end` 模式下直接返回。`wait_for_save` 用已经录下的 event 把满块放进队列。两处都不调用 SDK，都不 `synchronize`。单测里若 `wait_for_save` 同步了设备，会直接失败。

`end` 与 `per_layer` 的差别只在 event 录在哪。`per_layer` 让 D2H 流等待整条计算流，于是本步之后已经排到计算流上的内核也会被这次拷贝等住。`end` 在 `wait_for_save` 开始时于计算流上录一个 event，D2H 流只等这个 event。计算流不等 D2H 流。下一 step 的前向可以继续排进计算流。

## 3. 七条主机线程

worker 的 `start` 拉起七条守护线程，名字是 `surdata-lookup`、`surdata-preload`、`surdata-h2d`、`surdata-d2h`、`surdata-store`、`surdata-commit`、`surdata-storeGC`。每条线程的循环体只做自己的队列，空闲时等自己的 `threading.Event`。唤醒是 `set`，不是调用对方的函数，也不是 `join`。

| 线程 | 只做 | 不做 |
| --- | --- | --- |
| lookup | 向前缀索引问命中长度，把结果写成相位 | 不拷贝，不等 GPU 步，不落盘 |
| preload | 向 SDK 提交异步读，读完由回调改相位 | 不在线程里 `event.wait` 这次读 |
| h2d | 看见 `PRELOADED` 就提交 H2D，并轮询完成 | 不写盘 |
| d2h | 从队列取出已算完的块，在 D2H 流上拷进 store 池 | 不调用 commit |
| store | 调用 `enqueue_save`，写完由回调还槽 | 不调用 `commit_kv_durable_prefix` |
| commit | 本请求的 D2H 和写队列都空、请求已结束时，做一次 durable commit | 不发起拷贝 |
| storeGC | 把 store 池里已落盘且钉住次数为 0 的槽收回到水位 | 不回收 preload 池，不唤醒 preload |

线程之间传递的是队列和相位，不是调用栈。lookup 得到 0、又知道另一条请求还可能发布同一前缀时，在自己的线程里最多再等 2 秒、每 50 毫秒重问一次。这段等待留在 lookup 线程，不占 GPU 步。

commit 能看见 store 和 D2H 是否还占着这一请求，这是发布条件，不是把 commit 串回 store 线程。store 写完只 `set` commit 的唤醒。commit 自己再看 `_writes_settled`：这一请求仍在写、仍在 D2H、或还没有结束，就返回，下一拍再看。store 线程因此不会停在索引发布上。

storeGC 只等 store 池的水位 event。preload 池腾出空槽时唤醒 preload。store 池腾出空槽时唤醒 d2h。两个池不共用一条回收路径。

诊断线程 `surdata-phase-diag` 只有配置的间隔大于 0 才启动。它打一行相位计数，不参与搬运。

## 4. 三条 CUDA 流

设备侧是三条流。

**计算流**是 vLLM 当前的流。前向和 event 录在这里。SurCache 不往这条流提交 `memcpy`。

**D2H 流**是 worker 上单独的 `torch.cuda.Stream`，叫 store 流。生产路径的保存走它：先 `wait_event`（`end`）或 `wait_stream`（`per_layer`），再对每一层连续段调用 `cudaMemcpyAsync`，kind 为 DeviceToHost，然后在这条流上录完成 event。d2h 线程用 `event.query` 看完成，不 `synchronize`。完成后块才进入 store 队列。计算流上没有对这条流的等待。

**H2D 流**在传输库里，`cudaStreamCreate` 得到 `g_stream_h2d`。传输库另外还有 `g_stream_d2h` 和两条 C++ 线程，分别只发一个方向的拷贝。一条流上的 `memcpy` 不会排进另一条流。H2D 因此不必等正在进行的 D2H 排空。生产上的保存使用上面的 torch store 流，不走传输库的 D2H 提交；命中后的拷回走传输库的 H2D 流。

传输库发拷贝时，若这一段带了生产者 event，就在**自己的**流上 `cudaStreamWaitEvent`，然后 `cudaMemcpyAsync`，再在自己的流上录完成 event。等待的是「这块字节已经写好」这一个 event。H2D 流不等 D2H 流，D2H 流也不等 H2D 流。完成与否由主机侧 `poll` 查询 event，查询的线程不因此停住计算流。

`cudaHostRegister` 在建池时做一次。store 池和 preload 池都是钉住的主机内存，拷贝才能异步落到固定地址。池的分配不在计算线程里做。d2h 线程分到槽才提交拷贝；暂时没有槽，活着的请求下一拍再试，已经结束的请求丢掉这块，GPU 块可以归还。丢弃发生在 d2h 线程，计算线程不站在那里等空槽。

## 5. 相位是交接，不是屏障

调度进程和 worker 进程打开同一块共享内存状态板。调度器分配槽号，worker 按槽号写相位。调度器问「还能命中多少 token」时只读板：

- 这一条已经 `PRELOADED`，就回答命中长度。
- 队列里还有别的已预加载、本步还没回答的请求，而这一条仍在查找或读盘，就回答「还不知道」。调度器跳过它，继续问后面已经就绪的。
- 正在 H2D，仍回答「还不知道」，直到拷贝完成或失败。

「还不知道」不是让调度线程去等 lookup 线程。它让这一条先让开，已经在 DRAM 里的请求先进入 H2D。查找窗口缺省 32，多出来的请求留在等待环上，800 条不会同时堵在一次 SDK 调用里。窗口的归还也在相位变化时发生，不在计算线程里等待。

GPU 块的归还有自己的条件：请求已经结束，并且这一代在待拷贝、已交出、正在 D2H、待写盘的集合里都没有条目。条件在 d2h 和 store 的完成路径上满足。`request_finished` 只记下结束和要保留的块，不在计算线程里等这些集合变空。

## 6. 仍然存在的顺序

互不等待不是没有数据依赖。下面这些顺序还在，而且只沿一个方向。

拷贝不能早于产生这块字节的前向。所以 D2H 流等计算 event，H2D 流等生产者 event。计算流不反等。

索引里要能查到前缀，commit 必须晚于这块的写回调。所以 commit 线程看写队列和 D2H 是否还占着这一请求。store 线程不反过来等 commit 返回。

同一条 CUDA 流内部仍然按提交顺序执行。拆开的是流与流之间，不是一条流内部的 `memcpy` 与它自己的完成 event。

`per_layer` 让 D2H 流等待整条计算流。这比 `end` 等得更多：本步之后才排上计算流的内核也会挡住这次拷贝的开始。计算流本身仍然不等 D2H。缺省可以是 `per_layer`；要把「只等本步」变成默认行为，用 `end`。

取消已经发出的 `cudaMemcpyAsync` 收不回来。取消只盖住还没 `issue` 的任务。已经在设备上的那一次会跑完，完成时按代际丢掉，不写成命中。

## 7. 范围

本文只写这个拆分。SurCache 的直连实现在本仓库。LMCache 路径是另一份 Apache-2.0 仓库。`surdata-sdk` 和 SurFile 不在这里。SDK 内部的 RPC 仍可能阻塞调用它的那一条线程；调用点放在 lookup、preload、store、commit 上，就不放在计算线程上。

七条线程和三条设备流是当前代码里的划分。诊断线程不算搬运。传输库的 D2H 流还在，生产保存不使用它。

## 8. 结论

SurCache 把 KV 离开 GPU、再回到 GPU 的路径拆成互不等待的流。计算线程只录 event 并入队。lookup、preload、h2d、d2h、store、commit、storeGC 各等自己的唤醒，不在对方的函数里收尾。计算流、D2H 流、H2D 流分开。剩余的等待是单向的：拷贝流等生产者 event，commit 等写完，计算流不等它们。
