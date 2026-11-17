# WhoChat

Windows 本地聊天辅助软件。读取由使用者选择的聊天内容，提供建议，由使用者审核并自行发送。

## 安装和首次使用

打开 [Windows 安装包下载页](https://github.com/bandiaoseng/WhoChat/releases/tag/v0.1.0-windows.1)，下载 WhoChat-Setup-0.1.0-x64.exe，双击并选择安装位置。支持 Windows 10/11 64 位，自带 Python、Qt 和 RapidOCR 模型，无需安装开发工具。

从桌面启动后，在设置页填写自己的 AI 服务地址、模型和 API Key，再按软件提示绑定自己的聊天账号。安装包不包含任何人的客户库、密钥或额度。个人电脑上的受控启动器和额度账本不属于该安装版；使用的是本机设置的服务，费用由使用者自己的服务账户承担。

数据默认位于 %LOCALAPPDATA%/WhoChat，可通过 WHOCHAT_DATA_DIR 指定其他绝对路径。重新安装同一版本或升级不会覆盖该目录，卸载也保留本机数据。不要同时运行两个版本访问同一份数据。

这是首个同事试用安装版。已在构建电脑的独立空白数据目录验证界面启动和合成图片 OCR，未完成另一台干净电脑及真实聊天业务全流程验收。

## 开发和构建

源码基于 2026-11-14 审核快照，新增独立安装入口。开发依赖见 pyproject.toml；构建说明见 packaging/README.md。原始来源记录见 SOURCE_SNAPSHOT.json，后续隐私检查规则见 UPLOAD_POLICY.md。
