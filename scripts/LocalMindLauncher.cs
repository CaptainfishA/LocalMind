using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

internal static class LocalMindLauncher
{
    [STAThread]
    private static int Main(string[] args)
    {
        var root = AppDomain.CurrentDomain.BaseDirectory;
        var smokeTest = args.Length == 1 && args[0] == "--smoke-test";
        var start = new ProcessStartInfo
        {
            FileName = Path.Combine(root, "runtime", "python.exe"),
            WorkingDirectory = root,
            Arguments = smokeTest ? @"-B -X utf8 runtime\smoke_windows.py" : "-B -X utf8 -m localmind",
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardError = true,
        };

        string error;
        try
        {
            using (var python = Process.Start(start))
            {
                error = python.StandardError.ReadToEnd();
                python.WaitForExit();
                if (python.ExitCode == 0)
                    return 0;
                if (string.IsNullOrWhiteSpace(error))
                    error = "退出代码 " + python.ExitCode;
            }
        }
        catch (Exception exception)
        {
            error = exception.Message;
        }

        if (smokeTest)
            Console.Error.WriteLine(error);
        else
            MessageBox.Show(error, "LocalMind 启动失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
        return 1;
    }
}
