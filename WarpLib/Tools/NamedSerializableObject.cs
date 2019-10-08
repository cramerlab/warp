using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Warp.Tools
{
    [Serializable]
    public class NamedSerializableObject
    {
        public string Name;
        public object[] Content;

        public NamedSerializableObject(string name, params object[] content)
        {
            Name = name;
            Content = content;
        }
    }
}
