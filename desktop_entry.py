import os, sys, json, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

def data_root(name, variable):
    value = os.environ.get(variable)
    path = Path(value) if value else Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / name
    if not path.is_absolute():
        raise ValueError("数据目录必须是绝对路径")
    path.mkdir(parents=True, exist_ok=True)
    temp = path / "temp"
    temp.mkdir(exist_ok=True)
    os.environ["TEMP"] = os.environ["TMP"] = str(temp)
    import tempfile
    tempfile.tempdir = str(temp)
    return path

def main():
    data = data_root("WhoChat", "WHOCHAT_DATA_DIR")
    os.environ["WHOCHAT_DATA_DIR"] = str(data)
    if "--self-test" in sys.argv:
        report = Path(sys.argv[sys.argv.index("--self-test") + 1])
        result = {"ok": False, "data": str(data), "python": sys.prefix}
        try:
            from whochat.app import create_app
            from whochat.services.bootstrap import build_services
            from whochat.ui.main_window import MainWindow
            from whochat.config import ConfigStore
            from PySide6.QtCore import QTimer
            from PIL import Image, ImageDraw
            import numpy as np
            from rapidocr import RapidOCR
            import win32crypt, psutil
            app = create_app()
            config = ConfigStore().load()
            assert not config.ai.api_key, "Self-test requires a new empty profile"
            services = build_services()
            window = MainWindow(services)
            window.show()
            sample = Image.new("RGB", (700, 180), "white")
            ImageDraw.Draw(sample).text((30,60), "WHOCHAT TEST 123", fill="black", font_size=45)
            engine = RapidOCR()
            recognition = engine(np.array(sample))
            assert recognition.txts, "Bundled OCR did not recognize synthetic text"
            from whochat.services.native_open_files import NativeFileReader, ReaderUnavailable
            events = []
            with NativeFileReader(timeout=5, diagnostics=events.append) as reader:
                try:
                    reader.read(os.getpid(), psutil.Process().create_time())
                except ReaderUnavailable:
                    pass
            assert any(event["code"] == "not_wechat" for event in events), "Native worker did not return the expected refusal"
            window.grab().save(str(report) + ".png")
            result.update(ocrText=list(recognition.txts), ui=True, noApiKey=True, nativeWorker=True)
            QTimer.singleShot(800, app.quit)
            app.exec()
            services.shutdown()
            result["ok"] = True
        except Exception:
            result["error"] = traceback.format_exc()
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0 if result["ok"] else 1
    from whochat.app import main as app_main
    return app_main()

if __name__ == "__main__":
    raise SystemExit(main())
