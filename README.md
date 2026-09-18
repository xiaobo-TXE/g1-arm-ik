# G1 单臂位置控制（逆解 + ROS 2 + Groot VLA）

用 URDF 正解算出末端执行器位置 → 给定目标点 → 逆解控制末端到达。

支持两种下发通路：**ROS 2 话题/服务**，以及 **ZMQ 6002 帧**（对接
[unitree_rl_lab](https://github.com/lizqwerscott/unitree_rl_lab) 的 Groot 控制器，
手臂由外部下发、下肢由本机策略驱动）。

---

# Packages

| 包 | 层 | 作用 |
|---|---|---|
| `g1_ik` | 算法层 | 正解 / 逆解 / URDF 解析 / 精简模型。**零 ROS 依赖** |
| `g1_arm_ik_core` | 算法层 | 控制逻辑（限速/爬坡/看门狗）+ ZMQ 帧协议 + 机器人后端 |
| `g1_arm_msgs` | 接口层 | `SolveIK` / `SetArmEnabled` 服务，`ArmIKStatus` 消息 |
| `g1_arm_ik_node` | 节点层 | `ik_node`（ROS 话题通路）、`vla_node`（ZMQ 6002 通路） |
| `g1_arm_control_node` | 节点层 | 唯一下发运动的节点，限速/爬坡/看门狗 |
| `g1_arm_bringup` | 集成层 | 状态桥接、TF 广播、URDF、配置、launch |

**分层是硬约束，不是风格选择。** 算法层（`g1_ik` + `g1_arm_ik_core`）不含任何 ROS
依赖，所以能在一台没装 ROS 的机器上跑 59 项离线测试。静态校验器里有一项会主动阻断
`rclpy` / `tf2_ros` / `unitree_sdk2py` 的导入再加载算法层——一旦有人破坏这个分层，
那一项立刻失败。

---

# Dependencies

| 组件 | 版本（实测） | 说明 |
|---|---|---|
| Python | **≥ 3.10** | 见下方说明 |
| pinocchio (`pin`) | 4.1.0 | 正解、URDF、限位 |
| casadi | 3.8.1 | **官方 wheel 自带 IPOPT**，无需单独安装 |
| numpy | ≥ 1.24 | |
| PyYAML | ≥ 6.0 | VLA 帧序列化 + 配置 |
| pyzmq | ≥ 25 | **只有 VLA 通路需要** |
| ROS 2 | Humble（Ubuntu 22.04） | 只有 ROS 通路需要 |

## ⚠️ Python 版本下限是 3.10

在 Python 3.9 上 `pip install pin` **会失败**：pinocchio 依赖 `coal`，`coal` 依赖
`cmeel-assimp >= 6.0.5`，而该包没有 3.9 的 wheel，pip 转去源码编译并报错。
这是实测踩到的，不要试图在 3.9 上装。

## pinocchio / casadi 不能走 rosdep

它们的依赖链（`pin → coal → cmeel-assimp`）不在 ROS 打包体系里。先用 conda 或 pip 装好，
再用 colcon 编 ROS 包：

```bash
# 方式 A（推荐，与系统 python 隔离，避免 eigenpy 符号冲突）
conda create -n g1arm python=3.10 -y && conda activate g1arm
conda install -c conda-forge pinocchio casadi numpy pyyaml -y
pip install pyzmq

# 方式 B（系统 python）
pip3 install --user pin casadi numpy pyyaml pyzmq
```

方式 B 的坑：系统里如果已有 apt 装的 `ros-humble-pinocchio` 或 `libeigenpy`，pip 版会报
`undefined symbol: _ZN7eigenpy9NumpyType7getTypeEv`。用前先
`dpkg -l | grep -i eigenpy` 确认。方式 A 没这个问题。

**先验证 IPOPT 可用**（最容易翻车的一步）：

```bash
python3 -c "import pinocchio,casadi; print(pinocchio.__version__, casadi.__version__)"
python3 -c "
import casadi; o=casadi.Opti(); x=o.variable(); o.minimize((x-3)**2)
o.solver('ipopt'); print('IPOPT OK ->', float(o.solve()))"
```

第二条打印 `IPOPT OK -> 3` 才继续。

---

# Build

## 1. 离线算法层（不需要 ROS）

仓库不带 Python 环境和工具链（都是机器本地的东西，约 1.2 GB）。一条命令重建：

```bash
./tools_setup.sh
```

它会把 `uv` + Python 3.12 + 全部依赖装进 `tools/` 和 `.venv312/`（都被 git 忽略），
**完全不动系统 Python**，最后自动验证 IPOPT 可用。

**已经有 Python ≥ 3.10 的话，不需要这个脚本**，直接用 venv：

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # ZMQ 通路还需要: pip install pyzmq
```

然后跑全部检查（59 项离线 + 35 项静态校验）：

```bash
.venv312/bin/python run_all_checks.py
```

## 2. ROS 2 包装层

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

`--symlink-install` 让 Python 包以软链安装，改代码不用重编。

**用 conda 的话，先激活 conda 再 source ROS**，顺序反了 ROS 的 python 会盖掉 conda 环境，
`import pinocchio` 就找不到了。

---

# Usage

## 1. 离线：只调算法，不碰机器人

```bash
.venv312/bin/python demo.py                    # 左臂，自动加载 g1_left_arm.yaml
.venv312/bin/python demo.py --side right       # 右臂
.venv312/bin/python demo.py --trajectory       # 附带轨迹连续性演示
```

演示流程：正解现在的位置 → 给目标点 → 逆解 → 打印误差 → 轨迹连续性 → 不可达目标的安全行为。

## 2. ROS 2 话题通路

### 2.1 仿真栈（mock 后端，无硬件无 GUI）

```bash
ros2 launch g1_arm_bringup arm_sim.launch.py
```

三个终端分别看状态、使能、发目标：

```bash
ros2 topic echo /g1/arm_control/status

ros2 service call /g1/arm_control/set_enabled \
  g1_arm_msgs/srv/SetArmEnabled "{data: true}"

# 左臂（y 为正这一侧）
ros2 service call /g1/arm_ik_node/torso_target \
  g1_arm_msgs/srv/SolveIK "{x: 0.10, y: 0.15, z: -0.10}"

# 右臂（y 为负这一侧）
ros2 service call /g1/arm_ik_node/torso_target \
  g1_arm_msgs/srv/SolveIK "{x: 0.10, y: -0.15, z: -0.10}"

> 左右臂在 `y` 上**按符号对称**：左臂在 `+y` 侧，右臂在 `-y` 侧。给错符号的点虽然往往
> "看起来合理"，但逆解会差几厘米。用 `demo.py --side <side>` 先看该侧的实际可达范围最稳妥。
```

**必须先使能**，`control_node` 默认 DISABLED。

### 2.2 真机（四步，从安全到放行）

```bash
# ① 只接状态、只发到话题，机器人自己的控制仍接管手臂
ros2 launch g1_arm_bringup arm_bringup.launch.py \
    side:=left backend:=topic relay_source:=dds relay_network_interface:=eth0

# ② 检查状态流
ros2 topic echo /g1/joint_states --once    # 29 个关节，数值应是真实姿态
ros2 topic echo /g1/arm/command

# ③ DDS 空跑：整条链路构建好，每次 write 都是 no-op
ros2 launch g1_arm_bringup arm_bringup.launch.py \
    side:=left backend:=dds relay_source:=dds arm_dds_dry_run:=true

# ④ 确认无误后放行真发
ros2 launch g1_arm_bringup arm_bringup.launch.py \
    side:=left backend:=dds relay_source:=dds arm_dds_dry_run:=false
```

## 3. ZMQ 6002 通路（Groot VLA）

机器人跑 Groot 控制器时用这条。**手臂走 ZMQ，下肢由 Groot ONNX 策略本机推理驱动**，
两条数据源在你的 C++ 里合并。

```bash
# 干跑：解算、组帧、打印，一个字节都不发
ros2 launch g1_arm_bringup arm_vla.launch.py

# 看将要发送的帧原文（不接机器人也能确认格式）
ros2 topic echo /g1_arm_vla_node/frame_preview

# 发目标（下例是左臂；右臂把 y 取负）
ros2 service call /g1_arm_vla_node/torso_target \
  g1_arm_msgs/srv/SolveIK "{x: 0.10, y: 0.15, z: -0.10}"

# 确认无误后放行真发
ros2 launch g1_arm_bringup arm_vla.launch.py \
    zmq_enabled:=true zmq_dry_run:=false zmq_robot_ip:=<机器人IP>
```

节点有**双重静默门**（`enabled=false` + `dry_run=true`），并可用 `set_enabled`
服务随时把 6002 端口交还给别的生产者。

**⚠️ 6002 独占**：不要同时跑第二个发送端。两个 PUSH 对同一个 PULL 会交错，
而接收端发现某帧不是 14 个手臂关节就整帧丢弃。

---

# Detail of Packages

## g1_ik — 算法层

### `urdf_parser.py`（218 行）
零依赖 URDF 解析器，只用标准库 `xml.etree`。提供 `parse_urdf()` → `UrdfModel`
（links / joints / 父子关系 / 声明顺序 / 根链接）、`Joint` 数据类、
数学工具 `rpy_to_matrix()`（URDF 约定 `R = Rz·Ry·Rx`）、`axis_angle_to_matrix()`（罗德里格斯）。

**为什么手写**：它是逆解的**独立校验基准**。pinocchio 和它用不同代码路径算出相同的正解，
两者才可信；共用同一个解析器就只是自己验证自己。

### `reduced_model.py`（249 行）
把 29 自由度的整机 URDF 抽取成**单臂 7 关节**的精简 URDF（以 `torso_link` 为根，
追加固定关节把 `L_ee` 放在偏移 0.05 m 处），写到系统临时目录。

三条理由：**确定性**（构型向量恰好 7 维、顺序固定）、**结构性冻结腰部**（符合"腰当刚性
基座"的决定，不靠代价项软约束）、**速度**（CasADi/IPOPT 从 29 变量降到 7）。

### `model.py`（178 行）
pinocchio 封装。提供数值正解、全 link 位姿、关节原点、限位余量、名字↔向量互转。

**启动时做三项断言**：pinocchio 的关节顺序 == 精简模型记录的顺序；限位一致；
`L_ee` frame 存在。任一条不成立就抛异常——**静默的关节顺序错位在类人机器人上意味着
肘部指令驱动腕部**。

### `fk.py`（194 行）
手写 numpy 正解，逆解的独立校验基准。`fk_all_links()` 利用 URDF 声明顺序即拓扑序的特性，
一次遍历算出所有 link 位姿（TF 广播用它）。`G1NumpyFK` 提供 torso 系 / pelvis 系查询和
`T_pelvis_torso` 换算。

**踩过的坑写在代码里**：`L_ee` 是精简模型里的固定关节，如果走完整链**再加一次偏移**，
偏移会被应用两次（实测 3.9e-2 m 偏差）。所以先走到 `*_wrist_yaw_link` 再显式加偏移。

### `ik.py`（823 行）
逆解核心：CasADi + IPOPT，位置-only。含 `IKConfig`、`IKResult`、`ArmIK`。

**三个被实测否掉的设计**（文档保留了测量数据）：

| 方案 | 结果 |
|---|---|
| 姿态正则放进目标函数 | IPOPT 打满迭代，残差 **14.9 mm** |
| 平滑项权重 0.5 | 拖进局部极小，残差 **14.6 mm** |
| "位置硬约束 + 姿态最小化" | 姿态最优点落在约束边界，内点法无法收敛 |

**最终方案**：单阶段纯位置 + 硬限位约束。姿态自然性靠**热启动**和
`nullspace_project()`（精确零空间投影，改姿态**完全不移动末端**，实测 4.09° 关节运动
对应 1.1e-4 m 末端位移）。

**多起点回退**：IPOPT 返回局部极小值时状态仍是 `Solve_Succeeded`。实测有目标从零位启动
时肩膀顶在限位上、差 **77 mm**，而换随机种子 80% 能解到 0.0000 mm。误差超阈值就换种子
重解（每解 ~2 ms），成功率 57/60 → **60/60**。另有 `clearly_unreachable()` 预判，
不可达目标从 13 次尝试/20 ms 降到 1 次/13.5 ms。

**双符号后端**：优先用 `pinocchio.casadi`；该构建没编译 CasADi 支持时（PyPI 的 `pin`
wheel 就没有），退回自己用 CasADi SX 手搓的符号链。两者一致到 1e-15。

另含 `WeightedMovingFilter` / `RateLimiter` 供控制回路做输出调理。

### `config.py`（178 行）
YAML 加载，**未定义键报错**（拼错的参数等于没写）。`_validate_ik_config()` 拒绝 13 种
非法值，比如 `control_rotation=True` 但 `rotation_weight=0`（会接受姿态目标然后忽略它）。

### `__init__.py`（205 行）
`G1Arm` 门面类：组装精简模型 + pinocchio 封装 + numpy 正解 + 逆解器。
`build_arm_ik()` 是单次调用工厂。

---

## g1_arm_ik_core — 控制逻辑与协议

### `control.py`（514 行）
`ArmCommandShaper` 按重要性做四件事：

1. **看门狗** —— 状态流超过 `state_timeout` 没更新就**停发命令、权重归零**。节点中途死掉时
   这是唯一让手臂停下的机制。
2. **逐关节限速** —— 每周期最多动 `v_max × dt`。
3. **校验** —— 拒绝 NaN/Inf、超限位、离当前姿态超过 `max_joint_step` 的目标，并计数。
4. **arm_sdk 交接爬坡** —— 权重 0→1 平滑，从运控策略手里接管。

**一个被测试逼出来的设计修正**：原来只有收到目标后才爬权重，导致"已使能但还没目标"的
窗口里运控仍完全控制手臂。改成**使能即接管并保持当前姿态**。

`ControlConfig.validate()` 拒绝 14 种危险配置，最危险的是 `state_timeout<=0`
（**静默关闭看门狗**）。`MockBackend` 一阶关节模型让整条链路无硬件也能跑，
积分用**真实控制周期**（否则改控制率会改变仿真动力学）。

### `robot_backends/__init__.py`（167 行）
`mock`（内部仿真）/ `topic`（**桥接点**，订阅后转发给你自己的接口）/ `dds`
（`unitree_sdk2py` 直写 `rt/arm_sdk`）。

`DDSBackend.weight_index=-2` 是 arm_sdk 通行约定，但**依固件版本可能不同**，
务必对照官方 example 确认。

### `vla_protocol.py`（311 行）
ZMQ 6002 帧构造器，纯 Python。两张**必须逐位对齐**的表：`ARM_SLOT_LEROBOT_NAMES`
（`kLeftShoulderPitch` … ）与 `ARM_SLOT_SDK_NAMES`（`left_shoulder_pitch_joint` … ）。

`assert_slot_alignment(names, side)` 在节点启动时校验逆解库的关节顺序与 slot 一致。
**按 side 校验是修出来的 bug**——早期版本只认左臂，`side=right` 时节点直接启动失败。

`VlaActionFrame.validate()` 复刻接收端全部规则：14 个必须齐、值有限且 `|q| ≤ 3.2`、
速度有限。`next_timestamp()` 保证**严格递增**（时钟返回同值时自动 +1e-6，
实测冻结时钟下发 5 帧全被接受）。`VelocityCommand.as_remote_axes()` 做逆映射
（接收端 `vx=ly, vy=-lx, wz=-rx`）。

### `zmq_bridge.py`（155 行）
PUSH socket + `enabled`/`dry_run` 双重门。

**CONFLATE 是这里最关键的一行**：PUSH socket 在**没有对端连接**时不丢消息，而是排队直到
发送缓冲填满，然后 `send()` **阻塞**（实测 6.3 ms/次，甚至挂死测试）。加 `ZMQ_CONFLATE`
后只保留最新一帧，缓冲永不填满，`send()` 降到 **0.36 ms/次**。对命令流来说丢旧帧正是
想要的行为。`SNDTIMEO` 作兜底。

### `ros_common.py`（151 行）
关节布局唯一来源（`G1_29DOF_JOINT_NAMES` + `arm_indices()` / `waist_indices()`）、
`extract_by_name()`（**任何一个关节缺失就返回 None 而不是补零**——补零会产生看着合法但
完全错误的姿态）、`find_robot_description()`。

---

## g1_arm_ik_node — 逆解节点

### `ik_node.py`（324 行）
ROS 话题通路的逆解节点。三个目标入口：`torso_target` 服务（torso 系、直接解）、
`pelvis_target` 服务（pelvis 系、自动读腰角换算）、`torso_target_pose` 话题（流式）。

**安全姿态：逆解失败时什么都不发**，控制节点继续保持上一帧（手臂停在原地）。
`status.position_error` 上报**最后一次尝试**的残差（包括失败那次），因为填 0 会被读成
"完美解"。`_solve_lock` 串行化求解——`MultiThreadedExecutor` 下两个回调会同时改同一份
热启动状态。

### `vla_node.py`（435 行）
ZMQ 6002 通路的节点。目标 → 逆解 → 14 关节 → YAML 帧 → 6002。

**故意不做爬坡和限速**：你的 `ArmBezierTrajectory` 已经从实测姿态平滑接管，1 kHz 还有
逐帧限速，再加一层只会打架。

两个关键设计：`_last_success` vs `_last_result`（发送用**最后收敛的**姿态，诊断用最后
尝试；失败的解算**不能**清空正在执行的合法目标）；不控制的那条臂填**实测位置**
（接收端强制要求 14 个值，填实测值既不会和本机控制器打架也不会让自由臂瞬移）。

---

## g1_arm_control_node — 运动下发节点

### `control_node.py`（295 行）
ROS 话题 / DDS 通路**唯一下发运动**的节点。过 `ArmCommandShaper` 后写进后端。
**默认 DISABLED**，必须显式调 `set_enabled`。它不跑逆解（只用 URDF 拿限位和速度上限），
所以不会和 `ik_node` / `vla_node` 重复解算。

---

## g1_arm_msgs — 接口定义

| 文件 | 内容 |
|---|---|
| `srv/SolveIK.srv` | 请求 `x/y/z` + 可选姿态 + 可选热启动；响应 `success` / 位置误差 / `q_solution[7]` / 限位余量 / 迭代数 / 耗时 / 多起点次数 / 状态 |
| `srv/SetArmEnabled.srv` | `bool data` → `bool success` + `string message`。**早期版本漏了 `---` 分隔符，rosidl 编译不过**——被静态校验器抓到 |
| `msg/ArmIKStatus.msg` | 诊断：`q_current`/`q_target`/`q_command`、`arm_sdk_weight`、`ramp_progress`、`target_error`、`position_error`、`ik_ok`、`watchdog_tripped`、`state`、`backend`、收发计数、拒绝原因 |

---

## g1_arm_bringup — 集成层

### `joint_state_relay.py`（173 行）
Unitree `/lowstate` → `sensor_msgs/JointState`。`source:=ros`（用 `unitree_ros2` 的话题）
或 `source:=dds`（用 `unitree_sdk2py` 自己开 DDS 订阅）。

按**位置**映射 `motor_state[i]` → `G1_29DOF_JOINT_NAMES[i]`。代码注释明确要求
**验证你的固件真的是按电机序号升序发布**，否则下游拿到错位姿态。

### `tf_broadcaster.py`（181 行）
用我们自己的正解广播整棵 TF 树，不需要 `robot_state_publisher`，也不需要 mesh
（跟 RViz 无关；存在的意义是让你用 `ros2 run tf2_tools view_frames` 检查关节状态对不对）。

`require_complete_state`（默认开）：**关节状态不完整时不发 TF**。否则只收到手臂关节时，
腿会被当作 0 位发布——一棵**看着完全正常但错误**的树。

### `configure_urdf.py`（167 行）
**可选工具**。把官方 URDF 里相对的 mesh 路径改写成 `package://` 形式，可
`--fetch-meshes` 下载 STL（约 110 MB）。**本方案的 IK、控制、TF 都只读关节树、不读 mesh**，
所以这个脚本是可选的。

### `launch/`

| 文件 | 参数（默认值） |
|---|---|
| `arm_vla.launch.py` | `side=left` `zmq_robot_ip=192.168.123.222` `zmq_port=6002` `zmq_enabled=false` `zmq_dry_run=true` |
| `arm_ik.launch.py` | `side=left` `backend=mock` `urdf_path=` `joint_state_topic=/g1/joint_states` `dds_network_interface=eth0` `start_enabled=false` |
| `arm_sim.launch.py` | `side=left` |
| `arm_bringup.launch.py` | `side=left` `backend=topic` `dds_network_interface=eth0` `arm_dds_dry_run=true` `relay_source=dds` `relay_network_interface=eth0` `start_tf=false` |

### `config/`

| 文件 | 给谁 |
|---|---|
| `vla_params.yaml` | `vla_node`。**`zmq.robot_ip` 必须改** |
| `ik_params.yaml` | `ik_node` |
| `control_params.yaml` | `control_node`。`dds.network_interface` 默认 `eth0`，按实际网卡改 |
| `g1_left_arm.yaml` / `g1_right_arm.yaml` | 算法层，被 `demo.py` 按 `--side` 加载 |

`control_params.yaml` 里 `mock.noise` 故意写成 `0.00001` 而不是 `1e-5`——PyYAML（YAML 1.1）
会把无符号指数和无小数点的尾数解析成**字符串**，rclpy 启动时会拒绝。

### `urdf/`
官方 Unitree G1 URDF 三份：`g1_29dof_rev_1_0`（29 自由度）、
`g1_29dof_with_hand_rev_1_0`（带手）、`g1_dual_arm`（仅双臂，做实验干净）。

---

## 测试与校验

| 文件 | 项数 | 覆盖 |
|---|---|---|
| `test/conftest.py` | — | pytest 路径 shim：让测试在**没有 ROS、也没 colcon build** 的情况下能 import 到库 |
| `test/test_fk.py` | 11 | 解析器、rpy 约定、轴角、**关节限位/轴逐个断言**、精简模型、`L_ee` 偏移几何、**pinocchio 正解 vs 手写 numpy 正解**（1e-16）、腰链、可达范围 |
| `test/test_ik.py` | 16 | 左右臂往返（各 60 次）、亚微米精度、**轨迹连续性**、限位永不越界（含随机不可达）、不可达安全失败、符号后端等价、耗时、配置校验 |
| `test/test_control.py` | 11 | **看门狗停发**、逐关节限速、爬坡单调有界、目标校验、mock 闭环收敛、失能降权重、配置校验（14 种非法值） |
| `test/test_vla_protocol.py` | 10 | 帧格式。核心是 **`MockGrootReceiver`——逐行复刻 C++ 的 `parse_packet()`**，含全部拒绝路径；**故意发 7 种坏帧验证全被拒**；**全链路：真实逆解 → 组帧 → 过接收端 → 反向喂回正解验证末端回到目标点** |
| `test/test_zmq_bridge.py` | 5 | **真实 socket 收发**（真实 PUSH → 真实 PULL，按 `bind tcp://*:6002` 拓扑）、三重门、中途开关、**无对端不阻塞** |
| `test/test_vla_node_logic.py` | 6 | 失败的解算**不清空**正在执行的合法目标、从未成功过就不发、速度透传与过期；AST 检查节点构造是否还在（防测试验证已死逻辑） |
| `ros2_ws/validate_workspace.py` | 35 | package.xml、msg/srv 语法、入口点、`data_files`、launch 函数、**ROS 分层未被破坏**、关节索引断言、参数类型与死参数、**YAML 数值陷阱**、消息字段名、**配置是否真被加载**、**文档引用是否存在** |
| `run_all_checks.py` | — | 一键跑上面全部，退出码 0 表示全绿 |

---

## 其他文件

| 文件 | 作用 |
|---|---|
| `run_all_checks.py` | 一键跑全部测试套件与静态校验，退出码 0 表示全绿。**最常用** |
| `demo.py` | 算法层端到端演示，不需要 ROS；`--side` 自动选对应配置 |
| `requirements.txt` | Python 依赖清单。注释里写明**版本下限 3.10** 及原因 |
| `.gitignore` | 排除生成的精简 URDF、`__pycache__`、colcon 的 `build/install/log`、本地工具链、按需下载的 mesh |
| `tools/` | 本地工具链（`uv` + Python 3.12），完全装在项目目录内，不动系统 |
| `g1_arm_msgs/CMakeLists.txt` | `rosidl_generate_interfaces` 生成三个接口；`package.xml` 必须声明 `<member_of_group>rosidl_interface_packages</member_of_group>`（静态校验器会检查） |
| `*/setup.py` / `*/package.xml` | 各 ROS 包的构建描述与入口点。`data_files` **逐个列出**文件（不用 glob），缺文件会在构建时就报错 |
| `*/resource/<包名>` | ROS 2 包的资源标记文件，`ament` 靠它索引包 |

---

# 接口一览

## ROS 2 话题与服务

| 接口 | 类型 | 说明 |
|---|---|---|
| `/g1/joint_states` | `sensor_msgs/JointState` | 29 关节，按 G1 电机序号命名 |
| `/g1/arm_ik_node/torso_target` | `g1_arm_msgs/SolveIK` | torso 系目标点 |
| `/g1/arm_ik_node/pelvis_target` | `g1_arm_msgs/SolveIK` | pelvis 系目标点 |
| `/g1/arm_ik/torso_target_pose` | `geometry_msgs/PoseStamped` | 话题版目标 |
| `/g1/arm_ik/joint_target` | `std_msgs/Float64MultiArray` | 7 维逆解结果 |
| `/g1/arm/command` | `std_msgs/Float64MultiArray` | 整形后的命令 |
| `/g1/arm_control/set_enabled` | `g1_arm_msgs/SetArmEnabled` | 使能/失能 |
| `/g1/arm_control/status` | `g1_arm_msgs/ArmIKStatus` | 控制诊断 |
| `tf` | — | `pelvis` → 整棵 URDF 树 |

## Groot VLA 通路（`vla_node`）

| 接口 | 类型 | 说明 |
|---|---|---|
| `/g1_arm_vla_node/torso_target` | `g1_arm_msgs/SolveIK` | torso 系目标 |
| `/g1_arm_vla_node/pelvis_target` | `g1_arm_msgs/SolveIK` | pelvis 系目标 |
| `/g1_arm_vla_node/torso_target_pose` | `geometry_msgs/PoseStamped` | 流式目标 |
| `/g1_arm_vla_node/set_enabled` | `g1_arm_msgs/SetArmEnabled` | 开关 6002 发送 |
| `/g1_arm_vla_node/frame_preview` | `std_msgs/String` | **将要发送的帧原文** |
| `/g1_arm_vla_node/status` | `g1_arm_msgs/ArmIKStatus` | 诊断 |
| `/g1/vla/velocity` | `geometry_msgs/Twist` | 透传速度 |
| `tcp://<ip>:6002` | ZMQ PUSH（YAML） | 发往 `RemoteCommandReceiver` |

## ZMQ 6002 帧格式

```yaml
cmd: action
action:
  kLeftShoulderPitch.q: -0.20
  # ... 共 14 个手臂关节，必须齐
  kRightWristYaw.q: -0.01
  remote.lx: 0.0      # 映射 vx=ly, vy=-lx, wz=-rx
  remote.ly: 0.0
  remote.rx: 0.0
  remote.ry: 0.0      # 未使用
timestamp: 12345.67   # 只需严格递增；数值大小无关
```

| 约束 | 违反后果 |
|---|---|
| 必须恰好 14 个手臂关节 | 帧被丢 |
| 值有限且 `\|q\| ≤ 3.2` | 帧被丢 |
| timestamp 严格递增（数值本身无关） | 帧被丢 |
| 发送间隔 < 0.3 s（`vla_timeout_`） | 速度被钳成 0（安全，但手臂 hold） |

**关于 timestamp**：接收端只用它做一件事——严格递增门槛
（`timestamp <= previous->timestamp` 判为 stale）。数值大小**完全无关**：新鲜度是拿
**接收端自己**的到达时间算的（`out.received = steady_clock::now()`，与 FSM 线程的
`steady_seconds()` 比较），所以发送端和机器人的时钟不需要对齐。因此节点默认用
**单调时钟**而不是 ROS 时钟——`use_sim_time` 或 `/clock` 发布者会让 ROS 时钟倒退，
而倒退一次就会被判 stale 丢帧。可用 `zmq.timestamp_source` 切换（`monotonic` / `ros`）。

**腿部不在这条帧里。** 下肢由 Groot ONNX 策略本机推理驱动，
`State_Groot::publish_targets()` 在 1 kHz 线程里把两者合并成一条 `LowCmd`。

---

# 坐标系约定

```
pelvis ──waist_yaw── waist_yaw_link ──waist_roll── waist_roll_link ──waist_pitch── torso_link
                                                                                      │
                     left_shoulder_pitch → roll → yaw → elbow → wrist_roll → pitch → yaw
                                                                                      │
                                                                     [固定偏移 0.05 m] → L_ee
```

1. **逆解的输入输出都在 `torso_link` 坐标系。** 这是"腰当刚性基座"的直接结果。
2. **腰的 3 个关节不解。** 只用于换算 `p_torso = T_pelvis_torso(waist_q)⁻¹ · p_pelvis`，
   `waist_q` 从 `/lowstate` 按 `[waist_yaw, waist_roll, waist_pitch]` 读。
3. **末端是虚拟坐标系。** `L_ee` 挂在 `*_wrist_yaw_link` 上，沿腕 yaw 轴偏移 0.05 m
   （与官方 `xr_teleoperate` 一致）。**装了夹爪必须改成夹爪真实 TCP**，否则 IK 算得对、
   抓得歪。
4. **左臂在 `+y` 侧、右臂在 `-y` 侧**（torso 系）。同一个 `y` 符号给错，逆解会差几厘米甚至
   判成不可达。实测：`(0.10, +0.15, -0.10)` 左臂误差 0.0000 mm、右臂 41.2 mm。

---

# 实测结果

| 指标 | 值 |
|---|---|
| 正解一致性（pinocchio vs 独立 numpy 实现） | **5.6e-16** |
| 逆解位置误差（160 次随机往返） | 最差 **0.006 mm**，中位数 **0.0000 mm** |
| 逆解成功率 | 左臂 **60/60**、右臂 **60/60** |
| 求解耗时 | 热启动 **0.30 ms**（p95 0.33）、冷启动 **0.75 ms** |
| 关节限位越界（含 82 次含随机不可达目标） | **0 次** |
| 轨迹跟踪（120 步，步长 0.24 mm） | 最大误差 **0.00064 mm** |
| ZMQ 帧发送（无对端） | **0.36 ms/次**，不阻塞 |
| 测试套件 | **94/94 通过**（59 离线 + 35 静态） |

---

# 上真机前必须知道

1. **`ee_offset` 默认 0.05 m 是官方示例值。** 装夹爪必须换成真实 TCP
   （`config/ik_params.yaml` 和 `config/vla_params.yaml` 两处）。
2. **`zmq.robot_ip` / `dds.network_interface` 要改**（默认 `192.168.123.222` / `eth0`）。
3. **腕部 pitch/yaw 只有 5 N·m**（肩/肘 25 N·m）。末端别挂重载、别做顶压动作。
   URDF 实测值，已写进测试断言。
4. **不做碰撞检测。** 只保证关节限位和位置精度。手臂收回、贴身体、举过头顶这些位形
   IK 有解但会撞自己。要碰撞检查得上 MoveIt2，或用 `arm.m.fk_all_frames(q)` 自己算连杆距离。
5. **torso 是浮动的。** 站立/行走时腰角在变，必须每周期从 `/lowstate` 更新腰角再换算目标。
6. **逆解失败时节点什么都不发**，手臂保持在上一帧。一定要判 `success` 再决定后续动作。
7. **进入 VLA 的第一次接管很慢**：`ArmBezierTrajectory` 用 URDF 速度上限的 **2%**
   （`arm_transition.velocity_scale`），走 1 rad 约 **67 秒**。这是设计如此；
   之后稳态换目标是毫秒级。
8. **6002 端口独占**，两个 PUSH 对同一个 PULL 会交错丢帧。

---

# 已知限制与待办

**本机已验证（94 项）**：精度、控制逻辑、ZMQ 帧格式（复刻 C++ 解析器逐条验证，
含 7 条拒绝规则）、ZMQ 真实 socket 收发、包结构 / msg 语法 / 入口点 / 文件登记 /
配置可达性、算法层在阻断 rclpy 下仍可导入。

**只能在 Humble 机器上验证**：

- `colcon build` 能否通过（本机无 ROS，无法编译）
- 节点能否真正启动、话题服务是否连通
- `rclpy` 运行时行为
- **和真实 C++ `RemoteCommandReceiver` 对接**
- 真机时序：控制循环是否稳定、DDS 延迟
- VLA 接管是否真被触发、行进中腰角更新下的末端精度

**尚未实现**：

- **到达检测 / 完成事件。** 目前只能在应用层轮询 `ArmIKStatus.target_error`
  （`max|q_command − q_current|`）判断是否到位；`SolveIK` 的 `success` 只表示
  *逆解收敛*，不表示*手臂到位*。若需要"等到位再执行下一步"的时序语义，
  需要加到达判定（关节空间或笛卡尔空间）与事件/阻塞服务。
- 碰撞检测（见上文第 4 条）。

---

# License

BSD-3-Clause
