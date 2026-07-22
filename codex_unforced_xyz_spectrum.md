# Codex 指令：本地生成 Lorenz-63 三方向无外力自然频谱

请在当前 Lorenz 项目中新增一个独立的 Python 脚本，用于在本地计算并可视化无外力 Lorenz-63 系统的 \(x,y,z\) 三个方向自然频谱。我会根据图形人工判断频谱峰值。

只完成无外力自然频谱计算与可视化。保持实现直接、清晰，不加入自动峰值检测、平滑、预设频率标注、正弦强迫、一阶响应或二阶响应。

## 1. 计算目标

对多条独立的无外力 Lorenz-63 轨迹，在丢弃初始过渡段后，对 \(x(t),y(t),z(t)\) 分别计算单边复 FFT。

对每个方向 \(q\in\{x,y,z\}\)，计算不同轨迹复 Fourier 系数的样本标准差：

\[
S_q(f)=
\sqrt{
\frac{1}{N_{\mathrm{traj}}-1}
\sum_{r=1}^{N_{\mathrm{traj}}}
\left|Q_r(f)-\overline Q(f)\right|^2
}.
\]

这里 \(Q_r(f)\) 是第 \(r\) 条轨迹在该方向上的复 FFT 系数。

最终分别绘制 \(S_x(f),S_y(f),S_z(f)\)，让我人工观察峰值位置。

## 2. Lorenz-63 设置

使用标准无外力方程：

\[
\dot x=10(y-x),
\]

\[
\dot y=28x-y-xz,
\]

\[
\dot z=xy-\frac83z.
\]

参数集中放在脚本顶部，方便修改：

```python
FS = 16.0
SPINUP_TIME = 30.0
EFFECTIVE_TIME = 4096.0
N_TRAJ = 128
SEED = 0
RK4_SUBSTEPS = 4
```

要求：

- 每条轨迹初值独立采样自 \(N(0,1)\)；
- 无外部强迫；
- 丢弃前 `SPINUP_TIME` 个时间单位；
- spin-up 后保留 `EFFECTIVE_TIME` 个时间单位；
- 输出采样频率为 `FS`；
- 默认 `N_TRAJ = 128`，用于本地运行；
- `N_TRAJ` 保持为容易修改的参数，我之后可以改为 256、512 或更大；
- 固定随机种子，保证结果可复现。

## 3. 本地积分方式

本任务只计划在本地运行。优先采用 NumPy 向量化的固定步长 RK4，同时推进全部轨迹，不需要 MPI、Dask、Slurm 或远端调度代码。

状态数组形状保持为：

```python
state.shape == (N_TRAJ, 3)
```

一个输出采样间隔为：

```python
sample_dt = 1.0 / FS
```

每个输出采样间隔内部使用：

```python
internal_dt = sample_dt / RK4_SUBSTEPS
```

进行 `RK4_SUBSTEPS` 次 RK4 更新。

只保存 spin-up 后的采样值。为控制内存，可以分别保存为：

```python
samples.shape == (N_TRAJ, 3, N_SAMPLES)
```

默认 `N_TRAJ = 128` 时该规模可接受。数据类型优先使用 `float64`，FFT 使用 `complex128`。

## 4. FFT 定义

对每条轨迹和每个方向使用单边复 FFT：

```python
Q = np.fft.rfft(samples, axis=-1) / n_samples

if n_samples % 2 == 0:
    Q[..., 1:-1] *= 2.0
else:
    Q[..., 1:] *= 2.0

freq = np.fft.rfftfreq(n_samples, d=1.0 / FS)
```

必须保留复 FFT 系数完成跨轨迹统计。保持“先计算复系数统计量，再得到标准差”，不要先对每条轨迹取绝对值后再计算普通标准差。

跨轨迹统计可直接写为：

```python
mean_Q = Q.mean(axis=0)

variance = (
    np.sum(np.abs(Q - mean_Q[None, ...]) ** 2, axis=0)
    / (N_TRAJ - 1)
)

std_Q = np.sqrt(np.maximum(variance, 0.0))
```

结果应满足：

```python
std_Q.shape == (3, n_frequency)
```

其中顺序为 `x, y, z`。

## 5. 可视化

生成两张图。

### 5.1 完整频率范围

文件名：

```text
spectrum_full.png
```

要求：

- 三个纵向子图；
- 依次绘制 \(S_x(f),S_y(f),S_z(f)\)；
- 使用对数横轴和对数纵轴；
- 排除 \(f=0\)；
- 三个子图共享横轴；
- 不进行平滑；
- 不自动寻找峰；
- 不添加红点或预设频率；
- 标题中注明方向；
- 横轴为 `Frequency`;
- 纵轴为 `Std. of complex FFT coefficients`。

### 5.2 局部频率范围

文件名：

```text
spectrum_zoom_0p5_4Hz.png
```

要求：

- 同样使用三个纵向子图；
- 频率范围限制为 `0.5–4.0 Hz`；
- 保留原始频谱，不平滑；
- 让我直接人工观察三个方向在该频段的峰值。

图中或图注注明：

- `N_TRAJ`;
- `FS`;
- `SPINUP_TIME`;
- `EFFECTIVE_TIME`;
- `RK4_SUBSTEPS`。

## 6. 输出管理

所有结果统一写入：

```text
outputs/unforced_xyz_spectrum/
├── spectrum_full.png
├── spectrum_zoom_0p5_4Hz.png
├── spectrum_data.npz
└── run_info.json
```

不要在仓库根目录生成缓存、图片或临时文件。

`spectrum_data.npz` 至少保存：

```python
freq
std_x
std_y
std_z
mean_fft_x
mean_fft_y
mean_fft_z
```

`run_info.json` 至少保存：

```text
N_TRAJ
FS
SPINUP_TIME
EFFECTIVE_TIME
N_SAMPLES
RK4_SUBSTEPS
SEED
elapsed_seconds
```

## 7. 脚本接口

脚本文件名使用：

```text
lorenz_unforced_xyz_spectrum.py
```

支持命令行覆盖主要参数，例如：

```bash
python lorenz_unforced_xyz_spectrum.py \
  --n-traj 128 \
  --effective-time 4096 \
  --rk4-substeps 4
```

参数至少包括：

```text
--n-traj
--fs
--spinup-time
--effective-time
--rk4-substeps
--seed
--output-dir
```

默认输出目录为：

```text
outputs/unforced_xyz_spectrum
```

## 8. 本地计算注意事项

正式的 `EFFECTIVE_TIME = 4096` 和较大的轨迹数仍会有一定计算量，因此先执行一个小规模 smoke test：

```bash
python lorenz_unforced_xyz_spectrum.py \
  --n-traj 8 \
  --spinup-time 5 \
  --effective-time 32 \
  --rk4-substeps 2 \
  --output-dir outputs/unforced_xyz_spectrum_smoke
```

smoke test 只确认：

- 脚本正常结束；
- 输出文件齐全；
- 三个方向的频谱数组维度正确；
- 两张图可以打开。

smoke test 完成后，不要自动执行正式计算。

建议的首次本地正式运行命令：

```bash
python lorenz_unforced_xyz_spectrum.py \
  --n-traj 128 \
  --spinup-time 30 \
  --effective-time 4096 \
  --rk4-substeps 4
```

## 9. 完成后的回复

完成后只说明：

1. 新增或修改了哪些文件；
2. smoke test 是否通过；
3. 正式运行命令；
4. 修改轨迹数的位置或参数；
5. 输出目录。
