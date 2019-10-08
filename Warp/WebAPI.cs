using Nancy;
using Nancy.Extensions;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.ServiceModel;
using System.ServiceModel.Web;
using System.Text;
using System.Threading.Tasks;
using System.Windows;

namespace Warp
{
    public class WebAPIModule : NancyModule
    {
        public WebAPIModule()
        {
            Get["/GetSettingsGeneral"] = _ => JsonConvert.SerializeObject(MainWindow.Options, Formatting.Indented);
            Get["/GetSettingsImport"] = _ => JsonConvert.SerializeObject(MainWindow.Options.Import, Formatting.Indented);
            Get["/GetSettingsCTF"] = _ => JsonConvert.SerializeObject(MainWindow.Options.CTF, Formatting.Indented);
            Get["/GetSettingsMovement"] = _ => JsonConvert.SerializeObject(MainWindow.Options.Movement, Formatting.Indented);
            Get["/GetSettingsGrids"] = _ => JsonConvert.SerializeObject(MainWindow.Options.Grids, Formatting.Indented);
            Get["/GetSettingsPicking"] = _ => JsonConvert.SerializeObject(MainWindow.Options.Picking, Formatting.Indented);
            Get["/GetSettingsExport"] = _ => JsonConvert.SerializeObject(MainWindow.Options.Export, Formatting.Indented);
            Get["/GetSettingsTomo"] = _ => JsonConvert.SerializeObject(MainWindow.Options.Tomo, Formatting.Indented);
            Get["/GetSettingsFilter"] = _ => JsonConvert.SerializeObject(MainWindow.Options.Filter, Formatting.Indented);

            Get["/GetSettingsAll"] = _ => JsonConvert.SerializeObject(new Dictionary<string, object>()
            {
                {"General", MainWindow.Options },
                {"Import", MainWindow.Options.Import },
                {"CTF", MainWindow.Options.CTF },
                {"Movement", MainWindow.Options.Movement },
                {"Grids", MainWindow.Options.Grids },
                {"Picking", MainWindow.Options.Picking },
                {"Export", MainWindow.Options.Export },
                {"Tomo", MainWindow.Options.Tomo },
                {"Filter", MainWindow.Options.Filter },
            }, Formatting.Indented);

            Get["/GetProcessingStatus"] = _ => MainWindow.IsPreprocessing ?         "processing" : 
                                              (MainWindow.IsStoppingPreprocessing ? "stopping" : 
                                                                                    "stopped");

            Post["/StartProcessing"] = _ =>
            {
                ((MainWindow)Application.Current.MainWindow).StartProcessing();
                return "success";
            };

            Post["/StopProcessing"] = _ =>
            {
                ((MainWindow)Application.Current.MainWindow).StopProcessing();
                return "success";
            };

            Post["/LoadSettings"] = _ =>
            {
                try
                {
                    dynamic RequestJson = JsonConvert.DeserializeObject(Request.Body.AsString());

                    string Path = RequestJson["path"];

                    if (File.Exists(Path))
                    {
                        if (MainWindow.IsPreprocessing || MainWindow.IsStoppingPreprocessing)
                            throw new Exception("Can't change settings while processing.");

                        MainWindow.OptionsLookForFolderOptions = false;

                        MainWindow.Options.Load(Path);

                        MainWindow.OptionsLookForFolderOptions = true;

                        return "success";
                    }
                    else
                    {
                        throw new Exception("File not found.");
                    }
                }
                catch (Exception exc)
                {
                    return "fail: " + exc.ToString();
                }
            };
        }
    }
}
