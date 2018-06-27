using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Warp
{
    public class IOQueue
    {
        private int CacheCapacity;

        private Dictionary<string, Image> Cache = new Dictionary<string, Image>();
        private Queue<string> CacheQueue = new Queue<string>();



        public IOQueue(int capacity)
        {
            CacheCapacity = capacity;
        }
    }
}
