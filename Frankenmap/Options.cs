using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommandLine;
using CommandLine.Text;

namespace Frankenmap
{
    class Options
    {
        [Option("angpix", Required = true, HelpText = "Pixel size (must be identical for all maps).")]
        public float AngPix { get; set; }

        [Option("diameter", Required = true, HelpText = "Particle diameter in Angstrom.")]
        public int Diameter { get; set; }

        [Option("smooth", Default = 4f, HelpText = "Smoothing radius in pixels.")]
        public float SmoothingRadius { get; set; }

        [Option("fscthreshold", Default = 0.143f, HelpText = "DANGER ZONE!")]
        public float FSCThreshold { get; set; }
    }
}
