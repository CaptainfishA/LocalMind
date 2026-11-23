<img src="docs/assets/localmind-logo.png" alt="LocalMind 标志" width="128" height="128">

# LocalMind

你是否经常要找一份文件，却记不清它存在哪个目录？

你是否只记得文档里的一句话，或者大概的主题，却忘了文件名？

LocalMind 是一款面向中文用户的 Windows 本地文件搜索工具。

## 主要功能

- 按文件名、正文关键词或模糊描述找文件。
- 索引常见 Office 文档、PDF 和图片，支持 OCR。
- 查看命中依据，预览或打开原文件。
- 监控所选目录，随文件变化更新索引。

受上游 WeFlow 和 WeFlow Alpha 的停更及维护调整影响，**微信文件查询与聊天记录检索暂不提供**。

## 安装与配置

### 安装 LocalMind

在 GitHub 仓库的 **Releases** 中下载 `LocalMind-<版本>-windows-x64.zip`，完整解压后双击
`LocalMind.exe`。运行包适用于 Windows 10/11 x64，内含 Python 3.13 和应用依赖，默认使用 CPU。
若启动时提示缺少 VC 运行库，请安装 [Microsoft Visual C++ x64 运行库](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)。
首次使用语义搜索和 OCR 时可能需要下载模型。

#### 从源码运行

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

若 Windows 提示启动程序被占用，先关闭 LocalMind，再执行安装。
源码环境使用 Python 3.13。使用 CUDA 时需要匹配的 PyTorch 运行时，CPU 环境默认关闭 AI 精排。

### 微信接入参考（历史配置）

以下安装和配置步骤适用于已有的、支持微信接入的 LocalMind 环境。
[Alpha v0.3.5 发布页](https://github.com/xunchahaha/alpha-releases/releases/tag/v0.3.5) · [本机 HTTP API 配置说明](https://github.com/xunchahaha/alpha-releases/blob/v0.3.5/docs/api/http-api.md)

1. 从发布页下载并安装 Alpha v0.3.5，然后在 Alpha 中连接自己的微信账号。
2. 进入「设置 → 接口与集成 → 本机 HTTP API」，确认端口，生成或填写自己的访问令牌（Token），再点击「保存并启动」。
3. 在 LocalMind 中填写本机地址和 Token。默认地址是 `http://127.0.0.1:5031`，如果改过端口，请使用实际的端口。

API 使用本机回环地址。LocalMind 只在当前进程内存中保留 Token，重启后需重新填写。

## 本地文件

1. 在“本地文件”中选择目录并建立索引，不建议直接索引整个系统盘。
2. 默认用“综合”搜索；已知文件名/编号可用“关键词”，模糊描述可对照“语义”。
3. 点击结果核对匹配依据与预览。Office 预览为文字/表格内容，不保证原始排版。
4. 选定监控目录的变化由增量监听和启动补扫处理；索引更新后按提示刷新搜索结果。

“自动时间”可把“上个月合同”等表达解释为时间筛选；若日期只是正文内容，可选“不限时间”。
查询缺少直接证据时，仍可能返回相似资料。
分数仅用于当前模式排序，不是正确概率，也不跨模式比较。扩大检索会重新召回排序，不是稳定分页。

## 数据与隐私

- 只索引主动选择的目录。文件、OCR 和搜索默认在本机处理。
- 索引默认保存在 `%LOCALAPPDATA%\LocalMind\localmind.db`，模型默认保存在 `%LOCALAPPDATA%\LocalMind\models`。
- 模型缓存和已有的附件缓存可以更换位置、查看占用并清理。更换位置后，旧模型仍保存在原目录。
- 清空索引会关闭目录监控、启动补扫和已有的微信自动同步。原文件和原聊天保留在原位置。
- 此前建立的聊天索引以明文保存在本机。微信消息的删除和撤回尚无法自动同步到已有索引。

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
localmind-index stats
```
