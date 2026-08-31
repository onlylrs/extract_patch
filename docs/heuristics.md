# Heuristics

每一个heuristic方法可以按照列表的形式排成pipe，按顺序fallback。每一个heuristic都有成功/失败指标。

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

自定义 fallback：

```yaml
heuristic_pipe:
  - double_circle
  - smear
  - serrated_outer_circle
  - density
```

未列入 pipe 的方法不会执行。参数只需覆盖需要修改的字段：

```yaml
heuristics:
  serrated_outer_circle:
    min_confidence: 0.60
```

建议先运行 `./run_extract.sh --preview ...`，直接检查 contour 和 `patches/` 中的随机
patch，再决定是否调整。
