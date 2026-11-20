# LocalMind 发布指南

## 平时迭代

- 向 `main` 提交变更或发起 Pull Request 时，GitHub Actions 在 Windows x64 上构建 Python 包、校验包元数据并安装 wheel 执行测试。
- 在 **Actions → Windows CI and Release → Run workflow** 选择 Git 分支，可以按需构建 Windows ZIP。下载运行记录中的 `localmind-release` 工件即可验收。

## 发布一个版本

1. 在 `pyproject.toml` 和 `src/localmind/__init__.py` 更新相同的版本号。稳定版用 `0.7.0` 这类格式，预发布用 `0.7.0a4`、`0.7.0b1` 或 `0.7.0rc1`。
2. 合并到 `main` 并确认 Windows CI 通过。在对应提交上新建同版本标签并推送。下面以 `0.7.0a4` 为例。

   ```powershell
   git switch main
   git pull --ff-only
   git tag -a v0.7.0a4 -m "LocalMind 0.7.0a4"
   git push origin refs/tags/v0.7.0a4
   ```

3. 标签流水线通过后，在 **Releases** 打开自动创建的草稿。草稿附带 Windows x64 ZIP、Python 源码包和 `SHA256SUMS.txt`。Alpha、Beta 和 RC 会标记为预发布。
4. 下载 ZIP 核对 SHA-256，完整解压并启动 `LocalMind.exe`。确认版本说明、改动和已知限制后，点击 **Publish release**。

核对下载文件时，将下列命令的结果与 `SHA256SUMS.txt` 中对应文件名的记录比较。

```powershell
Get-FileHash -Algorithm SHA256 .\LocalMind-0.7.0a4-windows-x64.zip
```

构建或上传中断时，可在 Actions 重跑失败的任务。草稿会复用，已发布版本的附件会保留。发布后要修正内容时，请提高版本号并打新标签。

## 首次建立 GitHub 仓库

本地旧仓库的 Git 历史含开发记录，即使这些目录已被 `.gitignore` 忽略，历史提交仍会随推送上传。用审查过的源码快照在独立目录初始化 `main`，再推到 GitHub 私有仓库试跑。首发前确认源码、许可证、作者信息和 Release 附件，再将选定的仓库公开。

当前工作目录里的 `v0.7.0a3` 指向早期提交。可以在新仓库中为审查后的首发提交创建同名标签，或者选用新的版本号。后续版本在 GitHub 仓库的正常提交历史上逐次发布。

## 本地构建

在 Windows x64、Python 3.13 环境下执行。Windows ZIP 的启动器使用系统自带的 .NET Framework C# 编译器构建。

```powershell
python -m pip install build twine
python -m build --outdir dist/release
python -m twine check --strict 'dist/release/*'
python scripts/build_windows.py dist/release
```

Windows ZIP 内含 CPython 和 wheel 声明的依赖，默认按 CPU 环境验证。构建会检查 Python 下载的 SHA-256，并实际启动 ZIP 内的入口程序，在临时数据库里验证运行库、窗口和图标。模型权重与 Alpha / WeFlow 由用户按需另行获取。Windows 启动器目前未做 Authenticode 签名。

PySide6 目前限制在 6.12 之前。Windows 回归在 6.12 上曾于 Qt 退出时崩溃。升级时同时验证全量回归和运行包启动。
