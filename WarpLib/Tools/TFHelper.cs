using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using TensorFlow;

using TF_Status = System.IntPtr;
using TF_SessionOptions = System.IntPtr;
using TF_Graph = System.IntPtr;
using TF_OperationDescription = System.IntPtr;
using TF_Operation = System.IntPtr;
using TF_Session = System.IntPtr;
using TF_DeprecatedSession = System.IntPtr;
using TF_Tensor = System.IntPtr;
using TF_ImportGraphDefOptions = System.IntPtr;
using TF_Library = System.IntPtr;
using TF_BufferPtr = System.IntPtr;
using TF_Function = System.IntPtr;

namespace Warp.Tools
{
    public static class TFHelper
    {
        public static readonly object[] DeviceSync = { new object(), new object(), new object(), new object(),
                                                       new object(), new object(), new object(), new object(),
                                                       new object(), new object(), new object(), new object(),
                                                       new object(), new object(), new object(), new object(),
                                                       new object(), new object(), new object(), new object(),
                                                       new object(), new object(), new object(), new object()};

        public static TFSessionOptions CreateOptions()
        {
            TFSessionOptions Options = new TFSessionOptions();
            
            //byte[][] Serialized = new byte[][]
            //{
            //    new byte[] { 0x32, 0x5, 0x20, 0x1, 0x2a, 0x1, 0x30 },
            //    new byte[] { 0x32, 0x5, 0x20, 0x1, 0x2a, 0x1, 0x31 },
            //    new byte[] { 0x32, 0x5, 0x20, 0x1, 0x2a, 0x1, 0x32 },
            //    new byte[] { 0x32, 0x5, 0x20, 0x1, 0x2a, 0x1, 0x33 },
            //    new byte[] { 0x32, 0x5, 0x20, 0x1, 0x2a, 0x1, 0x34 },
            //    new byte[] { 0x32, 0x5, 0x20, 0x1, 0x2a, 0x1, 0x35 },
            //    new byte[] { 0x32, 0x5, 0x20, 0x1, 0x2a, 0x1, 0x36 },
            //    new byte[] { 0x32, 0x5, 0x20, 0x1, 0x2a, 0x1, 0x37 },
            //    new byte[] { 0x32, 0x5, 0x20, 0x1, 0x2a, 0x1, 0x38 },
            //    new byte[] { 0x32, 0x5, 0x20, 0x1, 0x2a, 0x1, 0x39 },
            //    new byte[] { 0x32, 0x6, 0x20, 0x1, 0x2a, 0x2, 0x31, 0x30 },
            //    new byte[] { 0x32, 0x6, 0x20, 0x1, 0x2a, 0x2, 0x31, 0x31 },
            //    new byte[] { 0x32, 0x6, 0x20, 0x1, 0x2a, 0x2, 0x31, 0x32 },
            //    new byte[] { 0x32, 0x6, 0x20, 0x1, 0x2a, 0x2, 0x31, 0x33 },
            //    new byte[] { 0x32, 0x6, 0x20, 0x1, 0x2a, 0x2, 0x31, 0x34 },
            //    new byte[] { 0x32, 0x6, 0x20, 0x1, 0x2a, 0x2, 0x31, 0x35 }
            //};
            byte[] Serialized = { 0x32, 0x2, 0x20, 0x1, 0x38, 0x1 };

            TFStatus Stat = new TFStatus();
            unsafe
            {
                fixed (byte* SerializedPtr = Serialized)
                    Options.SetConfig(new IntPtr(SerializedPtr), Serialized.Length, Stat);
            }

            return Options;
        }


        // extern TF_Session * TF_LoadSessionFromSavedModel (const TF_SessionOptions *session_options, const TF_Buffer *run_options, const char *export_dir, const char *const *tags, int tags_len, TF_Graph *graph, TF_Buffer *meta_graph_def, TF_Status *status);
        [DllImport(NativeBinding.TensorFlowLibrary)]
        static extern unsafe TF_Session TF_LoadSessionFromSavedModelOnDevice(TF_SessionOptions session_options, LLBuffer* run_options, string export_dir, string[] tags, int tags_len, TF_Graph graph, string device, TF_Status status);

        /// <summary>
        /// Creates a session and graph from a saved session model
        /// </summary>
        /// <returns>On success, this populates the provided <paramref name="graph"/> with the contents of the graph stored in the specified model and <paramref name="metaGraphDef"/> with the MetaGraphDef of the loaded model.</returns>
        /// <param name="sessionOptions">Session options to use for the new session.</param>
        /// <param name="runOptions">Options to use to initialize the state (can be null).</param>
        /// <param name="exportDir">must be set to the path of the exported SavedModel.</param>
        /// <param name="tags">must include the set of tags used to identify one MetaGraphDef in the SavedModel.</param>
        /// <param name="graph">This must be a newly created graph.</param>
        /// <param name="metaGraphDef">On success, this will be populated on return with the contents of the MetaGraphDef (can be null).</param>
        /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
        /// <remarks>
        /// This function creates a new session using the specified <paramref name="sessionOptions"/> and then initializes
        /// the state (restoring tensors and other assets) using <paramref name="runOptions"/>
        /// </remarks>
        public static TFSession FromSavedModel(TFSessionOptions sessionOptions, TFBuffer runOptions, string exportDir, string[] tags, TFGraph graph, string device, TFStatus status = null)
        {
            if (graph == null)
                throw new ArgumentNullException(nameof(graph));
            if (tags == null)
                throw new ArgumentNullException(nameof(tags));
            if (exportDir == null)
                throw new ArgumentNullException(nameof(exportDir));
            var cstatus = TFStatus.Setup(status);
            unsafe
            {
                var h = TF_LoadSessionFromSavedModelOnDevice(sessionOptions.handle, runOptions == null ? null : runOptions.LLBuffer, exportDir, tags, tags.Length, graph.handle, device, cstatus.handle);

                if (cstatus.CheckMaybeRaise(status))
                {
                    return new TFSession(h, graph);
                }
            }
            return null;
        }

        [DllImport(NativeBinding.TensorFlowLibrary)]
        public static extern unsafe void TF_FreeAllMemory();
    }
}
