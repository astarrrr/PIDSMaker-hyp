# Hyperbolic Temporal 全流程说明

## 1. 入口与配置解析
- 运行命令示例：`python pidsmaker/main.py hyperbolic_temporal THEIA_E3 --wandb`
- `main.py` 读取系统配置 `config/hyperbolic_temporal.yml`。
- `config/hyperbolic_temporal.yml` 继承 `orthrus_non_snooped`，并覆写：
  - `training.encoder.used_methods: hyperbolic_temporal`
  - `training.encoder.hyperbolic_temporal.*`（双曲时序模型参数）
  - `batching.intra_graph_batching.used_methods: edges`（避免 TGN 依赖）

## 2. 数据流水线（与其他系统一致）
按主流程顺序执行：
1. `construction`：按 `construction.time_window_size`（分钟）把事件流切成图快照。
2. `transformation`：图变换（本配置默认 `none`）。
3. `featurization`：训练节点文本嵌入（如 word2vec）。
4. `feat_inference`：给验证/测试图补齐嵌入。
5. `batching`：把图切成训练批次（本配置用 `edges`）。
6. `training`：训练编码器+目标头。
7. `evaluation`：计算异常分数并评估。
8. `triage`：可选告警后处理。

## 3. HyperbolicTemporal 编码器结构
实现文件：`pidsmaker/encoders/hyperbolic_temporal.py`

编码器按 HTGN 思路实现了四部分：
- `HGNN`：图注意力层（GAT）在切空间进行空间聚合。
- `HGRU`：GRUCell 在切空间进行时序状态更新。
- `HTA`：历史窗口注意力，融合过去 `window_size` 个时间状态。
- `HTC`：时序一致性约束，惩罚相邻时刻状态突变。

同时使用 HCC 风格的映射：
- `logmap0`：双曲空间 -> 切空间
- `expmap0`：切空间 -> 双曲空间
- 双曲球投影：保证表示保持在流形内部

## 4. 模型前向细节（单批次）
`Model.embed()` 调用编码器时会传入 `x / edge_index / t`，编码器内部流程：
1. 用边时间 `t` 聚合出节点时间统计，计算时间衰减 gate。
2. 把 gate 注入节点特征后做 HGNN 空间聚合，得到当前切空间表示 `h_tan_now`。
3. 读取历史状态（队列）做 HTA，得到历史上下文。
4. 用 HGRU 融合当前表示和历史上下文。
5. 通过 `expmap0` 投回双曲空间，得到节点嵌入 `h`。
6. 计算 HTC 正则：`aux_loss`（若存在历史状态）。
7. 更新历史状态队列（长度受 `window_size` 限制）。

## 5. 训练损失如何组成
- 主损失来自 `training.decoder.used_methods` 对应 objective（如 `predict_edge_type`）。
- 在 `pidsmaker/model.py` 中，若编码器返回 `aux_loss`，会在训练模式下加入总损失：
  - `total_loss = objective_loss + aux_loss`
- 因此 `HTC` 约束会真实参与反向传播。

## 6. 关键超参数与含义
在 `training.encoder.hyperbolic_temporal` 下：
- `num_layers`：HGNN 层数。
- `curvature`：双曲曲率参数（球半径相关）。
- `window_size`：HTA 关注的历史时间步数（w）。
- `htc_weight`：HTC 正则权重。
- `attention_temperature`：历史注意力温度。

常被混淆的参数：
- `construction.time_window_size`：每个图快照覆盖多少分钟（数据离散粒度）。
- `training.encoder.hyperbolic_temporal.window_size`：模型回看多少个历史快照（记忆长度）。

## 7. 与论文思路的对应关系
`docs/hyperbolic-temporal-network-embedding__read.org` 提到：
- HGNN + HGRU + HTA + HTC 主干
- 在双曲流形下进行时空建模

当前实现是工程化版本：
- 保留了上述四件套和双曲映射主线；
- 在现有 PIDSMaker 架构下复用 objective/decoder/evaluation 体系。

## 8. 运行建议
- 标准运行：
  - `python pidsmaker/main.py hyperbolic_temporal THEIA_E3 --wandb`
- 若没有 wandb 网络或权限：
  - `WANDB_MODE=offline python pidsmaker/main.py hyperbolic_temporal THEIA_E3 --wandb`
- 若 artifact 路径权限受限：
  - 增加 `--artifact_dir ./artifacts`

## 9. 当前已知环境依赖
- 需要可访问 PostgreSQL（默认 host 常为 `postgres`）。
- 若报 `could not translate host name "postgres"`，是数据库网络/容器 DNS 问题，不是模型实现问题。
