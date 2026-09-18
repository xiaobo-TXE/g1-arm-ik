# ROS 2 工作空间

**文档已统一到仓库根目录的 [`../README.md`](../README.md)。**
包结构、依赖、编译、运行、接口、坐标系约定、实测结果、真机注意事项全部在那里。

本文件只保留 ROS 2 特有的、根 README 里没有展开的部分。

---

## 编译

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

`--symlink-install` 让 Python 包以软链安装，改代码不用重编。

## 环境顺序（最容易出错的一点）

用 conda 装 pinocchio 的话，**必须先激活 conda，再 source ROS**：

```bash
conda activate g1arm                    # 先
source /opt/ros/humble/setup.bash       # 后
colcon build --symlink-install
```

顺序反了 ROS 的 python 会盖掉 conda 环境，`import pinocchio` 就找不到了。

## 验证编译前的静态检查

本工作空间带一个静态校验器，在**没有 ROS 的机器上**也能跑，用来抓那些
"要等到 colcon build 才暴露"的错误：

```bash
python3 validate_workspace.py
```

它检查 35 项：package.xml 完整性、msg/srv 语法（含 `---` 分隔符）、入口点是否指向
真实存在的函数、`data_files` 引用的文件是否存在、launch 函数、ROS 参数类型与死参数、
**YAML 数值陷阱**、消息字段名与 `.msg` 定义一致、**配置文件是否真的被加载**、
以及**算法层是否被混入 ROS 依赖**（会主动阻断 rclpy 再加载 core）。

## ROS 2 版本

只用 **Humble / Ubuntu 22.04**。官方 `unitree_ros2` 实测仅支持
Foxy/Ubuntu 20.04 与 Humble/Ubuntu 22.04，**没有 Jazzy / 24.04 支持**。
