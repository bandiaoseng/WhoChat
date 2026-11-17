using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;
class Launcher {
  [STAThread] static int Main(string[] args) {
    try {
      string folder=AppDomain.CurrentDomain.BaseDirectory;
      bool smoke=args.Length==2 && args[0]=="--self-test";
      string exe=Path.Combine(folder,"python",smoke?"python.exe":"pythonw.exe");
      string entry=Path.Combine(folder,"app","desktop_entry.py");
      string command="-E -s -B \""+entry+"\"";
      if(smoke) command+=" --self-test \""+args[1].Replace("\"", "")+"\"";
      ProcessStartInfo start=new ProcessStartInfo(exe,command);
      start.UseShellExecute=false;start.CreateNoWindow=true;start.WorkingDirectory=folder;
      start.EnvironmentVariables.Remove("PYTHONHOME");start.EnvironmentVariables.Remove("PYTHONPATH");
      Process child=Process.Start(start);
      if(smoke){child.WaitForExit();return child.ExitCode;}
      return 0;
    } catch(Exception ex) { MessageBox.Show("无法启动，请重新安装软件。\n"+ex.Message,"启动失败");return 1; }
  }
}
