# Codex 指令：修改实验 1 的自然频谱找峰方法

只修改实验 1：无外力 Lorenz-63 自然频谱分析。不要修改正弦强迫实验。

主要文件：

- `code/lorenz_sine/natural_spectrum.py`
- `scripts/analyze_spectrum_peak_significance.py`
- `configs/spectrum_server.json`

## 修改要求

1. 分别在 \(x\)、\(y\)、\(z\) 的跨 seed 平均 Welch PSD 上自动找峰。

2. 正式峰表不再使用归一化后的 `combined_psd`。它可以保留作辅助图，但不能作为主要候选峰来源。

3. 每个自动峰保存：

   - coordinate
   - rank
   - frequency
   - omega
   - bin
   - PSD value
   - prominence
   - width

4. 候选峰按 prominence 排序，并设置合理的最小峰距，避免同一个宽峰附近重复选点。

5. 删除当前在 `2.60–2.66` 区间内无条件选择最大 PSD 点的逻辑。

6. 若仍需检验 \(2.63\)，直接使用固定频率 `2.63` 的最近频率 bin，并标记为 `predefined_target`，不要标记为自动峰。

7. 使用固定且可复现的 discovery/test seed split：discovery seeds 只负责定峰，test seeds 只负责显著性检验。

8. 保留现有峰—背景统计量：

   \[
   D_r=\log(P_{\mathrm{peak}}/P_{\mathrm{background}})
   \]

   并对 test seeds 做单侧 t 检验和坐标内 Holm 校正。

9. 直接复用已有 spectrum run 中保存的 `freqs`、`psd_seed`、`psd_mean`，不要重新运行 Lorenz 积分。

10. 输出至少包括：

   ```text
   tables/detected_peaks_by_coordinate.csv
   tables/predefined_target_tests.csv
   figures/coordinate_psd_peaks.png
   figures/peak_significance.png
   ```

11. 图中三行分别画 \(x,y,z\) 平均 PSD，每个坐标只标自己的自动峰；自动峰和预设目标使用不同符号。

12. 横轴写 `frequency (Lorenz time unit^-1)`，不要写 Hz。

13. 暂停使用 `scripts/analyze_spectrum_harmonic_bands.py` 的结果作为正式自然频率或后续强迫频率来源。

修改完成后，用已有正式 spectrum run 做一次纯后处理，检查 \(z\) 的自动找峰结果中是否出现约 \(1.32\) 和 \(2.63\)。
