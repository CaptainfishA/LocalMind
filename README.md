# LocalMind

面向中文 Windows 用户的本地 AI 文件与微信资料搜索 Demo。

通过文件正文、聊天上下文或模糊描述找资料，核对命中依据，再预览、打开原文件。
默认本地处理，只索引主动选择的目录和会话，不修改原文件。微信依赖本机 Alpha / WeFlow；不支持 QQ。

## 当前能力

- 本地文档、PDF 和图片解析/OCR；关键词、语义、综合三种搜索模式。
- 微信消息与附件检索；可分别使用正文、聊天或全部线索，按会话、发送者、日期筛选。
- 结果显示实际证据、聊天来源与文件状态，支持文件预览、打开、打开目录和复制路径。
- 只读聊天时间线、引用/回复归组、已有结构化转发子视图；语义主题整理为可选实验功能。
- 目录增量监控、启动补扫、扫描取消；微信范围确认、批量续建和可选定时更新。
- 本地模型缓存、自定义缓存位置与安全清理、CPU/CUDA embedding、可选 CUDA/ONNX/CPU 精排。
- 常驻 SearchService/FAISS、跨文件批量 embedding、每次查询耗时与降级原因。

部分合并转发和附件受 Alpha / WeFlow 的接口能力限制。查询缺少直接证据时，仍可能返回相似资料。

## 安装与启动

在 GitHub 仓库的 **Releases** 中下载 `LocalMind-<版本>-windows-x64.zip`，完整解压后双击
`LocalMind.exe`。运行包适用于 Windows 10/11 x64，内含 Python 3.13 和应用依赖，默认使用 CPU。
若启动时提示缺少 VC 运行库，请安装 [Microsoft Visual C++ x64 运行库](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)。
语义搜索和 OCR 首次使用时可能需要下载模型。微信功能需要另行安装并配置本机 Alpha / WeFlow。

### 从源码运行

使用独立 Conda 环境。项目依赖要求以 `pyproject.toml` / `environment.yml` 为准。

```powershell
conda env create -f environment.yml
conda activate localmind-demo
localmind
```

已有环境只更新项目代码，不下载或更换依赖：

```powershell
conda activate localmind-demo
python -m pip install --no-deps --no-build-isolation --no-index -e .
localmind
```

若 Windows 提示启动程序被占用，先关闭 LocalMind，再执行安装。无需关闭微信或 Alpha。
Python 为 3.13；CUDA 需要匹配的 PyTorch 运行时，不是运行软件的前提。
首次使用未缓存模型可能下载；已有缓存优先本地加载。CPU 环境默认关闭 AI 精排。

## 本地文件

1. 在“本地文件”中选择目录并建立索引，不建议直接索引整个系统盘。
2. 默认用“综合”搜索；已知文件名/编号可用“关键词”，模糊描述可对照“语义”。
3. 点击结果核对匹配依据与预览。Office 预览为文字/表格内容，不保证原始排版。
4. 选定监控目录的变化由增量监听和启动补扫处理；索引更新后按提示刷新搜索结果。

“自动时间”可把“上个月合同”等表达解释为时间筛选；若日期只是正文内容，可选“不限时间”。
分数仅用于当前模式排序，不是正确概率，也不跨模式比较。扩大检索会重新召回排序，不是稳定分页。

## 微信资料

1. 在 Alpha / WeFlow 中连接自己的账号，手动开启本地 HTTP API，保持回环监听。
2. LocalMind 填写 Token 后点击“一键建立索引”；默认地址为 `http://127.0.0.1:5031`。
3. 确认账号、日期和会话范围后才读取消息。首次默认已识别的单人会话、群聊及近 30 天。
4. 可按名称筛选、全选/取消/反选；标签和按成员选群依赖接口能力，不可用时保留手动选择。
5. 搜索“微信附件”，按需要选择“全部线索 / 文件正文 / 聊天线索”。

暂停后的“继续建立索引”沿用原任务范围及截止时间；完成后更新所选索引以追上新消息。
批量自动更新默认关闭，开启后每 5 分钟更新已确认会话；新增会话不会自动加入。
取消勾选不删除已有索引，重启后需要重新输入 Token。不要将 Token 放进截图、日志或聊天。

附件打开与正文搜索是两个步骤：

- 在微信高级设置指定微信文件目录，或“关联本地文件”；没有可核验身份的同名候选需要确认。
- 文件路径确认后可直接打开；要搜索文件内容，再执行“索引附件正文”或批量正文索引。
- 单独索引附件不会自动把所在目录加入监控。需要持续跟踪时另行添加该目录。
- Alpha 返回 `media.not_found` 时，可在 Alpha 另存后关联或使用本地微信文件目录；不能凭消息恢复失效文件。
- 原消息跳转、任意媒体内联播放不作保证。聊天附近补拉是临时只读视图，不自动写入索引。

支持离线导入 ChatLab JSON/JSONL；不要选择微信加密数据库或 WeFlow 自有格式 JSON。
当前 Alpha 能力以实际目录和响应为准，公开文档不等于本机一定支持。

## 数据与隐私

- 默认索引：`%LOCALAPPDATA%\LocalMind\localmind.db`；模型：`%LOCALAPPDATA%\LocalMind\models`。
- 模型和按需取得的附件缓存可更换位置、查看占用并清理；更换位置不自动迁移旧模型。
- 清空索引关闭文件监控、启动补扫及微信自动同步，不删除用户原文件或原聊天。
- 聊天索引明文存储，删除索引不是磁盘安全擦除；微信删除/撤回尚不自动传播到本地快照。
- Token 只保留在当前进程内存；接口只允许本机回环 HTTP，不走系统代理、不跟随重定向。
- 不提取客户端密钥、不解密微信数据库、不自动扫描全盘、不监听屏幕。
- 默认不上传文件、OCR、查询或聊天。真实验证报告不保存正文、文件名、身份或 Token。

## 许可证

LocalMind 的自有代码采用 [GNU Affero General Public License 第 3 版](LICENSE)（`AGPL-3.0-only`）。
随附的桌面图标保留[原作者的版权与许可声明](src/localmind/desktop/assets/icons/LICENSE)。
第三方依赖、模型及用户另行安装的 Alpha / WeFlow 各自适用其许可。

## 开发与验证

```powershell
python -m pytest
python -m compileall src tests
python benchmarks/preview_desktop.py --interactive
```

合成界面预览使用独立临时索引，关闭后清理临时数据。版本打包与发布步骤见[发布指南](RELEASING.md)。

命令行示例：

```powershell
localmind-index scan "D:\Some Folder"
localmind-index search "年度预算" --no-rerank
localmind-index search "同事交接时发的巡检安排" --scope chat_files --evidence chat --mode hybrid --no-rerank
localmind-index stats
```
