using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommandLine;
using CommandLine.Text;

namespace Noise2Map
{
    class Options
    {
        [Option("observation1", Required = true, HelpText = "Relative path to a folder containing files with the first observation of the objects (e.g. first half-maps).")]
        public string Observation1Path { get; set; }

        [Option("observation2", Required = true, HelpText = "Relative path to a folder containing files with the second observation of the objects (e.g. second half-maps). Names of the files must match those of the first observation.")]
        public string Observation2Path { get; set; }

        [Option("denoise_separately", Default = false, HelpText = "If true, both observations will be denoised separately in the end. If false, their average will be denoised.")]
        public bool DenoiseSeparately { get; set; }

        [Option("old_model", Default = "", HelpText = "Name of the folder with the pre-trained model. Leave empty to train a new one.")]
        public string OldModelName { get; set; }

        [Option("dont_flatten_spectrum", Default = false, HelpText = "Don't flatten the spectrum of the maps beyond 10 Angstrom to sharpen them. Pixel size must be specified for flattening.")]
        public bool DontFlatten { get; set; }

        [Option("overflatten_factor", Default = 1f, HelpText = "Overflattening (oversharpening) factor in case a flat spectrum isn't enough. 1.0 = flat")]
        public float Overflatten { get; set; }

        [Option("angpix", Default = -1f, HelpText = "Pixel size used for spectrum flattening.")]
        public float PixelSize { get; set; }

        [Option("mask", Default = "", HelpText = "Relative path to a common mask for all maps. It can be used for spectrum flattening and map trimming.")]
        public string MaskPath { get; set; }

        [Option("upsample", Default = 1f, HelpText = "Upsampling factor, likely needed if resolution is close to Nyquist. 1.0 = no upsampling")]
        public float Upsample { get; set; }

        [Option("lowpass", Default = -1f, HelpText = "Low-pass filter to be applied to denoised maps (in Angstroms).")]
        public float Lowpass { get; set; }

        [Option("keep_dimensions", Default = false, HelpText = "If true, the denoised result will be downsampled to its original pixel size, and padded to the original box size.")]
        public bool KeepDimensions { get; set; }

        [Option("mask_output", Default = false, HelpText = "Masks the denoised maps with the supplied mask. Requires keep_dimensions to be enabled.")]
        public bool MaskOutput { get; set; }

        [Option("iterations", Default = 600, HelpText = "Number of iterations. 600–1200 for SPA half-maps, 10 000+ for raw tomograms.")]
        public int NIterations { get; set; }

        [Option("batchsize", Default = 4, HelpText = "Batch size for model training. Decrease if you run out of memory. The number of iterations will be adjusted automatically.")]
        public int BatchSize { get; set; }

        [Option("gpuid_network", Default = 0, HelpText = "GPU ID used for network training.")]
        public int GPUNetwork { get; set; }

        [Option("gpuid_preprocess", Default = 1, HelpText = "GPU ID used for data preprocessing. Ideally not the GPU used for training")]
        public int GPUPreprocess { get; set; }
    }
}
