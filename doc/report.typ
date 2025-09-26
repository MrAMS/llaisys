#set text(font: ("Noto Serif CJK SC"))
#import "@preview/lilaq:0.5.0" as lq

#set heading(numbering: "1.")

#align(center, text(17pt)[
  *项目报告*
])
#align(center)[
  2025年夏季InfiniTensor大模型与人工智能系统训练营 \
  Qijia Yang 中科院计算所
]

= 概述

- 项目地址：#link("https://github.com/MrAMS/llaisys")[github.com/mrams/llaisys]
- 完成情况：
  - 完成*CPU推理优化*（项目1），包括：SIMD指令、OpenMP多线程、分块(Cache Blocking)优化
  - 完成*构建AI聊天机器人*（项目3），包括：实现Gumbel-Max采样、移植网页UI、基于WebSocket实现实时流式输出
  - 完成*多用户推理服务*（项目4），包括：支持多用户请求调度、连续批处理、支持前缀匹配的KV-Cache池
  - *额外*单独使用LLAISYS提供的算子，实现了Python版本的Qwen2完整推理
  - *额外*完成了*Paged Attention/linear*算子和Paged KV Cache管理，通过内存分页有效解决了不同请求长度导致的内存碎片化问题

= CPU推理优化

== OpenMP

对几乎所有算子进行了OpenMP并行化

== SIMD

首先尝试直接使用x86 SSE2/SSE3 SIMD指令集实现了linear算子的SIMD向量化加速(#link("https://github.com/MrAMS/llaisys/commit/78939cd7dbc5593bfd82b6a93c96aa423bfeb93f")[commit-7893])，然后使用compiler vector extension进行了改写，不依赖于特定指令集，便于跨平台(#link("https://github.com/MrAMS/llaisys/commit/128cbdf918ea6626d6e9fccfe8680378cf4b3f07")[commit-128c])

== Cache Blocking

利用Cache特点，对linear算子中矩阵进行分块运算，提高Cache命中率从而提高性能

== Linear算子性能优化分析

测试了大小为$(512, 4096) * (4096, 4096)$、类型为`float32`、带bias的linear算子，结果如下：

#grid(
  columns: 2,
  gutter: 2em,
  align: top,
  figure(
    lq.diagram(
      xaxis: (
        ticks: ("None", "+OMP", "+SIMD", "+Blocking")
          .map(rotate.with(-45deg, reflow: true))
          .map(align.with(right))
          .enumerate(),
        subticks: none,
      ),
      ylabel: [Normalized],
      lq.bar(
        range(4),
        (1, 5936.54999/845.95023, 5936.54999/350.85551, 5936.54999/210.22395)
      )
    ),
    caption: [各种优化策略下相对性能]
  ),
  figure(
    table(
      columns: (auto, auto),
      inset: 10pt,
      align: horizon,
      table.header(
        [*优化策略*], [*耗时* (ms)],
      ),
      [None], [5936.54999],
      [OMP], [845.95023],
      [OMP+SIMD], [350.85551],
      [OMP+SIMD+Blocking], [210.22395]
    ),
    caption: [各种优化策略下linear算子耗时]
  ),
)

// None
// out (512, 4096), x (512, 4096), w (4096, 4096), bias True, dtype <f32>
// Torch time: 189.59108 ms 
// LLAISYS time: 5936.54999 ms

// OMP
// out (512, 4096), x (512, 4096), w (4096, 4096), bias True, dtype <f32>
// Torch time: 56.05230 ms 
// LLAISYS time: 845.95023 ms

// SIMD
// out (512, 4096), x (512, 4096), w (4096, 4096), bias True, dtype <f32>
// Torch time: 50.84760 ms 
// LLAISYS time: 350.85551 ms

// Cache
// out (512, 4096), x (512, 4096), w (4096, 4096), bias True, dtype <f32>
// Torch time: 59.00004 ms 
// LLAISYS time: 210.22395 ms



= 构建 AI 聊天机器人

== Gumbel-Max采样

实现了支持Temperature参数的Gumbel-Max随机采样算子

见代码`src/sampler/gumbelmax`

== 网页UI

移植了开源项目#link("https://github.com/ChristophHandschuh/chatbot-ui")[chatbot-ui] (#link("https://github.com/MrAMS/llaisys/commit/bb386f6fdda1202324d802f8925d09f4b86a1b7c")[commit-bb38])

见代码`webui/chatbot-ui`

#figure(
  image("imgs/webui1.png", width: 80%),
  caption: [
    网页UI欢迎界面
  ],
)

#figure(
  image("imgs/webui2.png", width: 80%),
  caption: [
    连续对话
  ],
)


== 实时流式输出

基于WebSocket实现了推理结果的实时流式输出(#link("https://github.com/MrAMS/llaisys/commit/bb386f6fdda1202324d802f8925d09f4b86a1b7c")[commit-bb38])

见代码`webui/backend`

= 多用户推理服务

== 多用户请求调度

使用双端队列`waiting`和`running`维护多用户请求，考虑到prefill阶段是compute-bound，而decoding阶段是memory-bound，使用以下调度策略：
- 每次调度运行至多`max_running_seqs`个用户的请求
- 当运行队列数量小于`max_running_seqs`时，优先调度执行prefill，当运行队列达到`max_running_seqs`时，批量执行decoding
- 当Paged KV Cache无法为新请求分配空间时，先尝试释放已完成请求的空间，再尝试释放`running`队尾的请求，并将此请求放在`wating`队首，若仍无法分配空间，则将本请求放至`wating`队尾

见代码`src/paged_cache.cpp/Scheduler::schedule()`

== 连续批处理

调度器每次调度运行至多`max_running_seqs`个用户的请求

见代码`src/paged_cache.cpp/Scheduler::schedule()`

== KV-Cache支持前缀匹配

支持前缀匹配的 KV-Cache 池，将KV Cache进行分块，使用`hashxx`库对每个Block及其前缀的tokens计算hash值，通过`map`维护每个hash对应block_id信息

见代码`src/paged_cache.cpp/Sequence::allocate()`

= Python版本推理

单独使用LLAISYS提供的算子，实现了Python版本的Qwen2完整推理

见代码`python/llaisys/models/qwen2_py.py`

= Paged KV Cache

- 仿照vLLM，将KV Cache进行分块，并且使用内存分页的方式，有效解决了不同请求长度导致的内存碎片化问题
- 修改支持了Paged Attention/linear/rope算子

见代码`src/paged_cache.cpp, src/ops/self_attention_paged, ...`



