# Heuristics

每一个 heuristic 方法可以按照列表的形式排成 pipe。默认
`heuristic_strategy: first` 保留按顺序 fallback、首个成功结果立即返回的行为。
`heuristic_strategy: arbitrate` 会运行全部方法，以最后一个 `density` 结果作为通用
前景参照；专用候选只有在置信度、前景召回和面积膨胀均通过安全门限后才会采用，
多个安全候选之间选择置信度最高者。

默认配置：有效密度

```yaml
heuristic_pipe: [density]
```

可用方法：

- `density`：通用细胞/染色密度分割，始终可作为兜底。
- `single_circle`：单个圆形载体或涂布区。
- `double_circle`：横向排列的双圆。
- `smear`：长轴明显的涂片区域。
- `smartcyto_circle`：完整保留 SmartCyto 的 single、double-texture、double-stain、
  double-edge、single-texture-Hough、single-texture-radial、enclosing refine 和安全判断。
  其特征图会先修复中性的暗色环，因此可降低泡沫/气泡边缘对纹理圆检测的干扰。
- `serrated_outer_circle`：先用受限径向拟合获得高召回核心，再在锯齿边缘内执行宽松密度
  扩展。最终 ROI 可以是圆、椭圆或不规则形状；硬性安全边界用于排除外层锯齿扫描轮廓。
- `qmh_center_circle`：QMH 专用的中央圆形涂布区；允许圆心偏移和圆被画布截断，并排除
  与左右边缘连通的黑色夹具。极稀疏样本会回退到 QMH 载体的几何先验。

自定义 fallback：

```yaml
heuristic_pipe:
  - double_circle
  - smear
  - serrated_outer_circle
  - density
```

需要同时支持多种未知载体时使用安全仲裁：

```yaml
heuristic_strategy: arbitrate
heuristic_arbitration:
  min_specialized_confidence: 0.70
  min_density_recall: 0.90
  max_density_area_ratio: 3.50
```

`qmh_center_circle` 是载体专用几何先验，不应放进无法确认 QMH 来源的通用配置。
`smear` 支持深色前景覆盖接近整个画布的 `full_field` 模式；该模式仍要求宽幅画布、
高 solidity 且轮廓至少接触三条画布边界。
`serrated_outer_circle` 通过外侧扫描环和内侧完整径向支撑自行执行安全检查，因此可设置
`density_guard: false`；这适用于 density 主要命中黑色扫描弧而非内侧稀疏样本的载体。

未列入 pipe 的方法不会执行。参数只需覆盖需要修改的字段：

```yaml
heuristics:
  serrated_outer_circle:
    min_confidence: 0.60
```

建议先运行 `./run_extract.sh --preview ...`，直接检查 contour 和 `patches/` 中的随机
patch，再决定是否调整。
