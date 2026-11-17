# Z1 抓取模型：推理主机部署说明

适用目标：Z1 Pro、原装夹爪、第三人称与腕部两台D435i。当前主机信息为Ubuntu20.04＋RTX4090，机械臂SDK已能工作。本文部署的是**推理服务**，不启动机械臂运动。

## 1. 部署边界与版本

```text
D435i×2 + Z1实测关节 → 控制程序 → WebSocket → 本服务
                       ↑                     ↓
                  本地动作执行器 ← 10×7原始关节目标
```

推理和控制可以在同一台电脑，也可以分开。先复用现有服务器推理也可；主机更换不应改变权重、统计量、图像处理或动作语义。

- OpenPI固定提交：`215abfb217dbac7d5f1273282331b9b1866c0479`，叠加仓库内推理补丁。
- LeRobot由该版本的`uv.lock`固定；不使用实验机上为训练数据读取修改的版本。
- 模型：π0.5、`gemma_2b_lora`＋`gemma_300m_lora`、32维内部动作、10步horizon、离散状态编码。
- 检查点：`reopening_mixed_20261110/200`。保存副本`aligned_suffix_control_20261111/0`来源相同，但导出后仍必须通过本仓库逐文件哈希，不凭文件夹名称判断。
- 对外动作：6个手臂关节＋夹爪，共7维绝对目标，弧度。无delta转换、无夹爪反号、无额外归一化。
- 固定指令：`抓起青色方块，抬高至少5厘米并保持2秒。`
- **当前无RTC。** 本服务不修改、限幅或执行动作；执行端须另接SDK、限位/限速、插值、超时和停止处理。
- 仿真独立验收为37/60；没有实机成功率，不保证仿真权重直接迁移后成功。

官方OpenPI列出4090可用于推理，但官方测试系统为Ubuntu22.04。本机20.04的完整环境尚未实际部署验证。优先使用下述独立Python环境，保留现有SDK环境；如出现系统库兼容问题再使用Ubuntu22.04容器，不要求重装控制电脑。GPU驱动兼容性以JAX实际检测结果为准，不要为安装本项目盲目更换现有驱动。

## 2. 准备代码和环境

本部署目录对应GitHub仓库`ChimDMY/pi0.5learning`；不要上传外层实验工作空间。推理主机上执行：

```bash
git clone https://github.com/ChimDMY/pi0.5learning.git
cd pi0.5learning
nvidia-smi
```

需要`git`、`git-lfs`、`ffmpeg`以及`uv`。Ubuntu安装基础依赖示例：

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs ffmpeg build-essential
git lfs install
```

按[uv官方说明](https://docs.astral.sh/uv/getting-started/installation/)安装uv，再执行：

```bash
bash scripts/setup.sh
```

脚本在`.runtime/openpi`建立固定版本源码和Python3.11独立环境，应用推理补丁，然后按原版锁文件安装依赖。不会覆盖SDK虚拟环境，不会启动GPU推理、训练或机械臂。

本仓库用`uv sync --frozen`，不要改成安装最新版OpenPI或直接换DROID配置。首次安装需要访问GitHub及Python依赖源，时间取决于网络与缓存。

若依赖下载中断，脚本保留目录且再次启动会拒绝覆盖。只有当日志确认源码checkout和补丁应用均已成功、失败发生在`uv sync`时，可继续：

```bash
cd .runtime/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --python 3.11
cd ../..
.runtime/openpi/.venv/bin/python verify_runtime.py
.runtime/openpi/.venv/bin/python -m unittest discover -s tests -v
```

如果源码准备阶段失败，应先检查原因，不能在未知版本源码上继续安装或重复叠加补丁。

## 3. 私下传输模型文件，不提交Git

源服务器上已选定的路径为：

```text
/media/share/dmy/openpi/checkpoints/pi05_z1_sim_lora/reopening_mixed_20261110/200/
```

只需要该目录的`params/`和`assets/`。它是完整Orbax推理参数，不是单独的LoRA adapter；不用传训练优化器，也不用再次下载整套pi05_base权重。

在目标主机仓库根目录执行，替换`USER@SOURCE_HOST`。以下命令保留源目录内容，不执行远端删除：

```bash
mkdir -p checkpoints/grasp_mixed200
rsync -av --progress \
  USER@SOURCE_HOST:/media/share/dmy/openpi/checkpoints/pi05_z1_sim_lora/reopening_mixed_20261110/200/params \
  checkpoints/grasp_mixed200/
rsync -av --progress \
  USER@SOURCE_HOST:/media/share/dmy/openpi/checkpoints/pi05_z1_sim_lora/reopening_mixed_20261110/200/assets \
  checkpoints/grasp_mixed200/
```

目标结构应为：

```text
checkpoints/grasp_mixed200/
├── params/
│   ├── _METADATA
│   ├── manifest.ocdbt
│   └── ...所有Orbax分片，保持原目录结构
└── assets/z1_sim_20261107/norm_stats.json
```

校验所有参数和归一化文件：

```bash
python3 check_artifacts.py checkpoints/grasp_mixed200
```

输出必须为`verified: true`。校验读取完整参数，磁盘较慢时需要等待。缺少分片、文件哈希不符、额外PyTorch权重等情况会停止，不能绕过检查。

还需要PaliGemma tokenizer。可联网时OpenPI首次运行自动下载；网络不通时，单独从源服务器复制：

```bash
mkdir -p cache/big_vision
rsync -av USER@SOURCE_HOST:~/.cache/openpi/big_vision/paligemma_tokenizer.model \
  cache/big_vision/paligemma_tokenizer.model
export OPENPI_DATA_HOME="$PWD/cache"
```

若源服务器设置了不同的`OPENPI_DATA_HOME`，按实际缓存路径复制。Tokenizer也不放进Git。

## 4. 首次启动：仅本机监听

先确认目标GPU空闲。下面的`--gpu 0`指**新推理主机自己的4090编号**，不表示可以占用旧服务器GPU0。若在共享服务器启动，必须明确选择空闲GPU或其UUID，不停止其他任务。

```bash
export OPENPI_DATA_HOME="$PWD/cache"
bash scripts/start.sh \
  --checkpoint "$PWD/checkpoints/grasp_mixed200" \
  --gpu 0 --host 127.0.0.1 --port 8000
```

启动时依次完成权重校验、源码身份校验、模型加载、一次合成输入预热，之后才开始监听。首次编译耗时可能较长，以`Ready`日志及HTTP检查为准，不以进程存在为准。

另开终端，在仓库根目录运行：

```bash
curl --fail --max-time 5 http://127.0.0.1:8000/healthz
.runtime/openpi/.venv/bin/python smoke_client.py --host 127.0.0.1 --port 8000
```

客户端必须返回`passed: true`、`action_shape: [10,7]`、`robot_moved: false`。默认使用黑图和零状态，仅证明模型可加载、协议及形状正确，不证明能抓取。此客户端有有限连接和接收超时，不会无限轮询。

若有真实双相机与关节观测，可以保存为NPZ：`overview`/`wrist`为240×320×3的uint8 RGB；`state`为7个实测弧度。再执行：

```bash
.runtime/openpi/.venv/bin/python smoke_client.py \
  --observation private/first_observation.npz --host 127.0.0.1 --port 8000
```

仍不下发动作。`private/`需要自行建立，文件默认不提交Git。

## 5. 控制电脑通过局域网访问

本机通信成功后，停止旧服务，再以推理主机的实际局域网地址监听。例如把下方`INFERENCE_LAN_IP`替换为实际地址：

```bash
bash scripts/start.sh \
  --checkpoint "$PWD/checkpoints/grasp_mixed200" \
  --gpu 0 --host INFERENCE_LAN_IP --port 8000
```

服务没有账户认证或TLS，仅供可信局域网；防火墙只允许控制电脑访问对应端口，不做公网映射。也可以保留127.0.0.1监听，通过SSH隧道访问。

控制电脑只运行客户端时，无需安装JAX或整个模型。可以用独立Python3.11环境安装固定版本客户端：

```bash
uv venv --python 3.11 .client-venv
uv pip install --python .client-venv/bin/python \
  'openpi-client @ git+https://github.com/Physical-Intelligence/openpi.git@215abfb217dbac7d5f1273282331b9b1866c0479#subdirectory=packages/openpi-client' \
  'numpy==1.26.4' 'websockets==15.0.1'
.client-venv/bin/python smoke_client.py --host INFERENCE_LAN_IP --port 8000
```

客户端需要本仓库的`smoke_client.py`和`contract.py`。首次先做这一条有限通信检查，不启动机械臂。不要把相机传输、GPU推理等待放进Z1 SDK需要持续运行的高频控制循环。

## 6. 控制端必须遵守的接口

请求通过OpenPI msgpack_numpy WebSocket发送，不能用普通JSON序列化大数组：

```python
{
    "schema": "z1_grasp_chunk_v1",
    "request_id": "每次请求唯一的字符串",
    "noise_seed": 812,  # 每次重规划由客户端生成并记录新的种子
    "prompt": "抓起青色方块，抬高至少5厘米并保持2秒。",
    "observation/image": overview_rgb,          # uint8[240,320,3]
    "observation/wrist_image": wrist_rgb,       # uint8[240,320,3]
    "observation/state": measured_q,           # float32[7]
}
```

- 图像是RGB、HWC；不是OpenCV默认BGR，不是深度，不是CHW或0–1浮点图。
- 两个D435i按序列号固定身份。先固定真实成像视角与裁剪规则，再变为320×240；不要把宽屏画面随意拉伸而宣称与仿真一致。服务内部复用原224尺寸模型预处理，控制端不要再次归一化。
- 状态顺序为joint1…joint6、gripper，必须来自反馈，不用上次命令代替实测。真机零位、正负号和夹爪角度到训练约定的映射仍待现场确认。
- 返回`actions`为10×7原始模型目标。包含`request_id`、推理耗时与记录编号。前5步对应200ms训练动作时长；客户端是否能按此节奏执行取决于实测端到端延迟。
- 模型输出偶尔可能越界。本服务只检查形状和有限性，**不把模型输出当成合法SDK目标**。必须由控制端做基于实机限制的校验与执行处理，不复制仿真驱动增益。
- 此协议一次返回整块，控制端管理动作队列；与旧仿真服务逐次返回单个`action`的协议不同。不要接旧客户端后假定它会自动取前5步。
- 每次请求明确记录noise_seed，用于复现该次推理。不要每200ms重复固定同一随机种子；这与旧服务内部连续随机流也不逐位相同，部署层未宣称完整轨迹等价。
- 训练目标中的开爪约−1rad、闭爪目标0rad，是训练坐标约定。夹住物体后的实测值不必为0；原装夹爪仍需验证SDK对应关系。

当前没有包含真机执行器，不能通过这个仓库直接进行自主抓取。部署完推理后，下一步是接入已有SDK程序、安装腕部相机、核对映射，再进行首次固定位置抓取。

## 7. 日志与检查点记录

每次启动创建独立`runs/时间_编号/`：

- `loaded_policy.json`：实际权重路径、检查点清单哈希、源码身份、GPU选择、预热结果、动作语义。
- `requests.jsonl`：请求编号、随机种子、时间与模型推理耗时。
- 每次请求的`.npz`：两路实际输入图、关节状态、噪声和原始预测块。

这些是每次重规划的观测，不是连续25Hz视频；执行后的关节反馈、实际下发目标和全程双视视频需要控制端记录。本服务不会假装已经保存真实机械臂轨迹。运行目录不自动删除，定期查看磁盘余量。

## 8. 常见问题

| 现象 | 处理 |
|---|---|
| `OpenPI source mismatch` | 使用独立固定提交与补丁，别指向实验机被持续修改的工作树 |
| 检查点哈希失败 | 核对是不是mixed200，是否完整复制params与assets；不要只复制manifest或norm |
| 找不到tokenizer/下载卡住 | 复制tokenizer缓存并设置OPENPI_DATA_HOME |
| `No JAX GPU device` | 检查nvidia-smi、GPU选择和JAX CUDA依赖；CPU导入通过不等于GPU可用 |
| `ModuleNotFoundError` | 确认使用`.runtime/openpi/.venv/bin/python`，不是SDK的Python |
| 显存不足 | 查看是否有其他任务占用，先停本服务；不杀其他进程，不随意切换模型结构 |
| 错误prompt、图像形状、状态维数 | 按第6节接口修正；不要改服务器校验去掩盖输入错误 |
| 通信成功但延迟大于200ms | 暂不按25Hz直接驱动；记录延迟后决定队列策略，不能让真机像仿真一样暂停物理时间 |

## 9. 单独上传GitHub

在这个目录内初始化仓库。先只生成本地提交；只有确认远端仓库及可见性后再推送：

```bash
git init
git add README.md DEPLOY.md VALIDATION.md OPENPI_LICENSE .gitignore .gitattributes \
  contract.py z1_config.py unitree_z1_policy.py check_artifacts.py \
  verify_runtime.py serve.py smoke_client.py scripts manifests patches tests
git diff --cached --stat
git diff --cached
git commit -m "Add pinned Z1 pi0.5 grasp inference deployment"
git branch -M main
git remote add origin https://github.com/ChimDMY/pi0.5learning.git
# 确认远端仓库及可见性后执行：git push -u origin main
```

不上传权重、日志、相机图片、私钥、密码或源服务器地址。仓库只包含源码与摘要清单；真实部署参数留本机。

## 10. 官方参考

- [OpenPI环境与推理要求](https://github.com/Physical-Intelligence/openpi#requirements)
- [OpenPI远程推理协议](https://github.com/Physical-Intelligence/openpi/blob/main/docs/remote_inference.md)
- [Unitree Z1 SDK](https://github.com/unitreerobotics/z1_sdk)
- [RealSense多相机示例](https://github.com/realsenseai/librealsense/blob/master/examples/multicam/readme.md)

官方链接用于说明接口和部署方式；本项目实际执行版本以清单和固定提交为准。
