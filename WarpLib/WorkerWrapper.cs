using System;
using System.Diagnostics;
using System.IO;
using System.IO.Pipes;
using System.Runtime.Serialization.Formatters.Binary;
using System.Threading;
using Warp.Sociology;
using Warp.Tools;

namespace Warp
{
    public class WorkerWrapper : IDisposable
    {
        bool IsAlive = true;

        public int DeviceID = 0;
        string UID = "";

        NamedPipeServerStream StreamSend;
        NamedPipeServerStream StreamReceive;

        NamedPipeServerStream StreamHeartbeat;
        Thread Heartbeat;

        BinaryFormatter Formatter;

        Process Worker;
        
        public WorkerWrapper(int deviceID)
        {
            DeviceID = deviceID;
            UID = Guid.NewGuid().ToString();

            StreamSend = new NamedPipeServerStream($"WarpWorker{UID}_out", PipeDirection.Out, 1, PipeTransmissionMode.Byte);
            StreamReceive = new NamedPipeServerStream($"WarpWorker{UID}_in", PipeDirection.In, 1, PipeTransmissionMode.Byte);

            StreamHeartbeat = new NamedPipeServerStream($"WarpWorker{UID}_heartbeat", PipeDirection.Out, 1, PipeTransmissionMode.Byte);
            Heartbeat = new Thread(new ThreadStart(() =>
            {
                while (IsAlive)
                {
                    StreamHeartbeat.WaitForConnection();
                    StreamHeartbeat.Disconnect();

                    Thread.Sleep(100);
                }
            }));
            Heartbeat.Start();

            Formatter = new BinaryFormatter();

            Worker = new Process
            {
                StartInfo =
                {
                    FileName = Path.Combine(Helper.PathToFolder(System.Reflection.Assembly.GetEntryAssembly().Location), "WarpWorker.exe"),
                    CreateNoWindow = false,
                    WindowStyle = ProcessWindowStyle.Minimized,
                    Arguments = $"{DeviceID} WarpWorker{UID} {Debugger.IsAttached}"
                }
            };
            Worker.Start();
        }

        #region Private

        bool SendCommand(NamedSerializableObject command)
        {
            StreamSend.WaitForConnection();

            Formatter.Serialize(StreamSend, command);

            StreamSend.Disconnect();

            return GetSuccessStatus();
        }

        bool GetSuccessStatus()
        {
            StreamReceive.WaitForConnection();

            NamedSerializableObject Result = (NamedSerializableObject)Formatter.Deserialize(StreamReceive);

            StreamReceive.Disconnect();

            if (Result.Name == "Success")
            {
                return (bool)Result.Content[0];
            }
            else
                throw new Exception("Received message of unknown type!");
        }

        public void Dispose()
        {
            if (IsAlive)
            {
                IsAlive = false;

                SendCommand(new NamedSerializableObject("Exit"));

                StreamSend.Dispose();
                StreamReceive.Dispose();
                StreamHeartbeat.Dispose();
            }
        }

        ~WorkerWrapper()
        {
            Dispose();
        }

        #endregion

        public bool Ping()
        {
            return SendCommand(new NamedSerializableObject("Ping"));
        }

        public void SetHeaderlessParams(int2 dims, long offset, string type)
        {
            if (!SendCommand(new NamedSerializableObject("SetHeaderlessParams",
                                                         dims,
                                                         offset,
                                                         type)))
                throw new Exception("Couldn't set headerless parameters!");
        }

        public void LoadGainRef(string path, bool flipX, bool flipY, bool transpose, string defectsPath)
        {
            if (!SendCommand(new NamedSerializableObject("LoadGainRef",
                                                         path,
                                                         flipX,
                                                         flipY,
                                                         transpose,
                                                         defectsPath)))
                throw new Exception("Couldn't load the gain reference!");
        }

        public void LoadStack(string path, decimal scaleFactor, int eerGroupFrames)
        {
            if (!SendCommand(new NamedSerializableObject("LoadStack",
                                                         path,
                                                         scaleFactor,
                                                         eerGroupFrames)))
                throw new Exception("Couldn't load the stack!");
        }

        public void MovieProcessCTF(string path, ProcessingOptionsMovieCTF options)
        {
            if (!SendCommand(new NamedSerializableObject("MovieProcessCTF",
                                                         path,
                                                         options)))
                throw new Exception("Couldn't fit the CTF!");
        }

        public void MovieProcessMovement(string path, ProcessingOptionsMovieMovement options)
        {
            if (!SendCommand(new NamedSerializableObject("MovieProcessMovement",
                                                         path,
                                                         options)))
                throw new Exception("Couldn't fit the movement!");
        }

        public void MovieExportMovie(string path, ProcessingOptionsMovieExport options)
        {
            if (!SendCommand(new NamedSerializableObject("MovieExportMovie",
                                                         path,
                                                         options)))
                throw new Exception("Couldn't export the movie!");
        }

        public void MovieExportParticles(string path, ProcessingOptionsParticlesExport options, float2[] coordinates)
        {
            if (!SendCommand(new NamedSerializableObject("MovieExportParticles",
                                                         path,
                                                         options,
                                                         coordinates)))
                throw new Exception("Couldn't export the particles!");
        }

        public void TomoProcessCTF(string path, ProcessingOptionsMovieCTF options)
        {
            if (!SendCommand(new NamedSerializableObject("TomoProcessCTF",
                                                         path,
                                                         options)))
                throw new Exception("Couldn't fit the CTF!");
        }

        public void TomoExportParticles(string path, ProcessingOptionsTomoSubReconstruction options, float3[] coordinates, float3[] angles)
        {
            if (!SendCommand(new NamedSerializableObject("TomoExportParticles",
                                                         path,
                                                         options,
                                                         coordinates,
                                                         angles)))
                throw new Exception("Couldn't export the particles!");
        }

        public void MPAPreparePopulation(string path)
        {
            if (!SendCommand(new NamedSerializableObject("MPAPreparePopulation",
                                                         path)))
                throw new Exception("Couldn't prepare population!");
        }

        public void MPARefine(string path, string workingDirectory, string logPath, ProcessingOptionsMPARefine options, DataSource source)
        {
            if (!SendCommand(new NamedSerializableObject("MPARefine",
                                                         path,
                                                         workingDirectory,
                                                         logPath,
                                                         options,
                                                         source)))
                throw new Exception("Couldn't perform MPA refinement!");
        }

        public void MPASaveProgress(string path)
        {
            if (!SendCommand(new NamedSerializableObject("MPASaveProgress",
                                                         path)))
                throw new Exception("Couldn't save MPA refinement progress!");
        }

        public void TryAllocatePinnedMemory(long[] chunks)
        {
            if (!SendCommand(new NamedSerializableObject("TryAllocatePinnedMemory",
                                                         chunks)))
                throw new Exception("Couldn't allocate requested chunks!");
        }
    }
}
