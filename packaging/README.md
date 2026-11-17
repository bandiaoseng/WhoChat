# Windows 安装包构建

采用独立 CPython 3.12.10 + NSIS，保留 Python 标准启动方式，兼容应用的原生辅助子进程。已检索 PyInstaller；本次没有采用，避免把依赖 sys.executable 的子进程改造成冻结进程协议。应用没有依赖系统 Python。

1. 在干净 Windows Python 3.12 环境安装 runtime-lock.json 中记录的依赖，准备 NSIS 与 .NET Framework 4 编译器。
2. 执行 packaging/prepare_runtime.py <工作目录>/python <根依赖...>。脚本只复制标准库和第三方 distribution RECORD 指定且哈希核对通过的文件，排除虚拟环境配置、测试缓存和用户数据。
3. 将应用源码放入 <工作目录>/app，入口为 desktop_entry.py。WhoChat 还需 rapidocr/models 的三个上游 OCR 模型，首次安装版核对其哈希与原审核版本一致。收藏助手 tools/ 中需 ffmpeg.exe、ffprobe.exe 及对应许可说明。
4. 编译 Launcher.cs 为对应软件名.exe，执行 build_installer.py --payload <工作目录> --output <安装包.exe> --makensis <makensis.exe>。输出目录不得位于 payload 内部。
5. 使用新的空白数据目录执行 软件名.exe --self-test <报告.json>，在隔离路径静默安装并比较 payload-manifest.json，验证升级保留数据，然后扫描待提交源码并发布。

用户数据不能放进 payload。所有测试和缓存须指定到充足空间的磁盘；本次构建和验证全部在 F 盘。首次安装包未签名，不应宣称已经完成干净机器/真实账号业务验收。

工具来源：
- https://github.com/python/cpython （PSF License，包含在 python/LICENSE.txt）
- https://github.com/NSIS-Dev/nsis （许可包含在 licenses/NSIS.txt）
- https://github.com/electron-userland/electron-builder （同一套已验证 NSIS 工具链）
- https://github.com/pyinstaller/pyinstaller （已评估，未运行）

第三方 Python 模块的原始许可保留在各 dist-info 或包目录中；版本清单见 runtime-lock.json。依赖代码和 DLL 保持独立文件，未加密或限制替换。
