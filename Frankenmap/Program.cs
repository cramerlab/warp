using CommandLine;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Warp;
using Warp.Tools;

namespace Frankenmap
{
    class Program
    {
        static void Main(string[] args)
        {
            Options Options = new Options();
            string WorkingDirectory;

            if (!Debugger.IsAttached)
            {
                Parser.Default.ParseArguments<Options>(args).WithParsed<Options>(opts => Options = opts);
                WorkingDirectory = Environment.CurrentDirectory + "/";
            }
            else
            {
                Options.AngPix = 1.05f;
                Options.Diameter = 200;
                Options.SmoothingRadius = 4;
                WorkingDirectory = @"H:\Composite_Sth1+NCP\";
            }

            Options.Diameter = (int)(Options.Diameter / Options.AngPix / 2) * 2 + 10;

            string[] MapPaths = Directory.EnumerateFiles(WorkingDirectory, "*half1*.mrc").ToArray();
            int3 Dims = new int3(400);

            Console.WriteLine($"{MapPaths.Length} maps found.\n");

            Directory.CreateDirectory(Path.Combine(WorkingDirectory, "localres"));
            Directory.CreateDirectory(Path.Combine(WorkingDirectory, "masks"));
            Directory.CreateDirectory(Path.Combine(WorkingDirectory, "composite"));

            for (int imap = 0; imap < MapPaths.Length; imap++)
            {
                Console.Write($"Calculating local resolution for {Helper.PathToNameWithExtension(MapPaths[imap])}... ");

                Image Map1 = Image.FromFile(MapPaths[imap]);
                Image Map2 = Image.FromFile(MapPaths[imap].Replace("half1", "half2"));

                Dims = Map1.Dims;

                Image Mask = Map1.GetCopy();
                Mask.Fill(1);
                Mask.MaskSpherically(Options.Diameter, 1, true);

                Image LocalRes = new Image(Map1.Dims);
                Image LocalB = new Image(Map1.Dims);

                int SpectrumLength = 40 / 2;
                int SpectrumOversampling = 2;
                int NSpectra = SpectrumLength * SpectrumOversampling;

                Image AverageFSC = new Image(new int3(SpectrumLength, 1, NSpectra));
                Image AverageAmps = new Image(AverageFSC.Dims);
                Image AverageSamples = new Image(new int3(NSpectra, 1, 1));
                float[] GlobalLocalFSC = new float[40 / 2];

                GPU.LocalFSC(Map1.GetDevice(Intent.Read),
                             Map2.GetDevice(Intent.Read),
                             Mask.GetDevice(Intent.Read),
                             Map1.Dims,
                             1,
                             Options.AngPix,
                             LocalRes.GetDevice(Intent.ReadWrite),
                             40,
                             Options.FSCThreshold,
                             AverageFSC.GetDevice(Intent.ReadWrite),
                             AverageAmps.GetDevice(Intent.ReadWrite),
                             AverageSamples.GetDevice(Intent.ReadWrite),
                             SpectrumOversampling,
                             GlobalLocalFSC);

                Mask.Dispose();
                Map2.Dispose();
                Map1.Dispose();

                Image LocalResInvSmooth = LocalRes.AsConvolvedGaussian(2f, true);
                LocalRes.Dispose();
                LocalB.Dispose();

                AverageFSC.Dispose();
                AverageAmps.Dispose();
                AverageSamples.Dispose();
                    
                LocalResInvSmooth.WriteMRC(Path.Combine(WorkingDirectory, "localres", $"{(Helper.PathToName(MapPaths[imap]))}_localres.mrc"), Options.AngPix, true);
                LocalResInvSmooth.Dispose();

                Console.WriteLine("Done.");
            }

            Console.WriteLine("");

            {
                Console.Write("Creating masks... ");

                float[][] LocalRes = Helper.ArrayOfFunction(i => Image.FromFile(Path.Combine(WorkingDirectory, "localres", $"{(Helper.PathToName(MapPaths[i]))}_localres.mrc")).GetHostContinuousCopy(), MapPaths.Length);
                float[][] Membership = Helper.ArrayOfFunction(i => new float[LocalRes[0].Length], LocalRes.Length);
                               

                for (int i = 0; i < LocalRes[0].Length; i++)
                {
                    float BestRes = 1000;
                    int BestID = 0;

                    for (int j = 0; j < LocalRes.Length; j++)
                    {
                        if (LocalRes[j][i] < BestRes)
                        {
                            BestRes = LocalRes[j][i];
                            BestID = j;
                        }
                    }

                    Membership[BestID][i] = 1;
                }

                Image[] Masks = Helper.ArrayOfFunction(i => new Image(Membership[i], Dims), LocalRes.Length);
                for (int i = 0; i < LocalRes.Length; i++)
                {
                    Image Convolved = Masks[i].AsConvolvedGaussian(Options.SmoothingRadius, true);
                    Masks[i].Dispose();
                    Masks[i] = Convolved;
                    Convolved.FreeDevice();
                }

                Console.WriteLine("Done.\n");
                Console.Write("Sewing together the frankenmap monster... ");

                Membership = Masks.Select(v => v.GetHostContinuousCopy()).ToArray();

                float[][] Halves1 = Helper.ArrayOfFunction(i => Image.FromFile(MapPaths[i]).GetHostContinuousCopy(), MapPaths.Length);
                float[][] Halves2 = Helper.ArrayOfFunction(i => Image.FromFile(MapPaths[i].Replace("half1", "half2")).GetHostContinuousCopy(), MapPaths.Length);

                float[] Composite1 = new float[LocalRes[0].Length];
                float[] Composite2 = new float[LocalRes[0].Length];

                for (int i = 0; i < LocalRes[0].Length; i++)
                {
                    float Sum1 = 0;
                    float Sum2 = 0;
                    float Samples = 0;

                    for (int j = 0; j < LocalRes.Length; j++)
                    {
                        float Weight = Membership[j][i];
                        Sum1 += Weight * Halves1[j][i];
                        Sum2 += Weight * Halves2[j][i];
                        Samples += Weight;
                    }

                    Composite1[i] = Sum1 / Samples;
                    Composite2[i] = Sum2 / Samples;
                }

                new Image(Composite1, Dims).WriteMRC(Path.Combine(WorkingDirectory, "composite", "composite_half1.mrc"), Options.AngPix, true);
                new Image(Composite2, Dims).WriteMRC(Path.Combine(WorkingDirectory, "composite", "composite_half2.mrc"), Options.AngPix, true);

                for (int i = 0; i < Membership.Length; i++)
                    Masks[i].WriteMRC(Path.Combine(WorkingDirectory, "masks", $"{(Helper.PathToName(MapPaths[i]))}_mask.mrc"), Options.AngPix, true);

                Console.WriteLine("Done.");
            }
        }

        private static void ClearCurrentConsoleLine()
        {
            int currentLineCursor = Console.CursorTop;
            Console.SetCursorPosition(0, Console.CursorTop);
            Console.Write(new string(' ', Console.WindowWidth));
            Console.SetCursorPosition(0, currentLineCursor);
        }
    }
}
