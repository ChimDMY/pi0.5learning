# pi0.5learning — Z1 Pro / π0.5 抓取推理部署

用于部署已微调的 Z1 抓取模型：**抓起青色方块，抬高至少5厘米并保持2秒**。

此目录可单独作为 GitHub 仓库。完整安装与运行步骤见 **[DEPLOY.md](DEPLOY.md)**。

已完成的检查及尚未验证的范围见[VALIDATION.md](VALIDATION.md)。

包含固定版本 OpenPI 的推理补丁、Z1 双相机输入变换、模型身份校验、推理服务、单次通信客户端和CPU接口检查。

- 输入：第三人称RGB、腕部RGB、6关节＋夹爪实测角度、固定抓取指令。
- 输出：10×7原始绝对关节目标，训练动作间隔40ms，既有仿真每次执行前5步。
- 权重：`reopening_mixed_20261110/200`；归一化和所有参数文件以SHA256固定。
- 实验背景：仿真独立验收37/60；这不是实机成功率。
- **本仓库只提供推理，不包含真机驱动、动作限幅、插值控制、急停或已完成的RTC实机实现。模型输出不能直接视为SDK可执行命令。**

权重、训练数据、相机图像、运行日志和凭据通过Git之外的渠道传输，`.gitignore`已排除常见路径。

```text
serve.py                   推理主机入口
smoke_client.py            单次通信/预测检查，不控制机械臂
z1_config.py               固定模型结构和预处理
unitree_z1_policy.py        六臂＋夹爪的输入/输出变换
contract.py                双RGB、7维状态和动作块协议
check_artifacts.py         检查权重/归一化哈希
verify_runtime.py          检查OpenPI源码身份
scripts/setup.sh           创建隔离推理环境
scripts/start.sh           启动推理服务
manifests/                 权重及源码身份，不含权重
patches/                   来自实际运行版本的必要推理源码差异
tests/                     轻量接口检查
```

OpenPI及其派生代码遵循[原始许可证](OPENPI_LICENSE)，第三方依赖及模型权重另遵循各自许可；本仓库不分发模型权重。补丁用于保留既有推理行为，不代表重新设计了π0.5架构。

上传前执行`git diff --cached --stat`和`git diff --cached`，确认提交范围只有此目录中的代码、说明、补丁和清单。不要在整个实验工作空间根目录执行`git add .`。
