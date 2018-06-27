﻿using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using Accord;
using Warp.Tools;

namespace Warp
{
    public class Star
    {
        Dictionary<string, int> NameMapping = new Dictionary<string, int>();
        List<List<string>> Rows = new List<List<string>>();

        public int RowCount => Rows.Count;
        public int ColumnCount => NameMapping.Count;

        public Star(string path, string tableName = "", int nrows = -1)
        {
            using (TextReader Reader = new StreamReader(File.OpenRead(path)))
            {
                string Line;

                if (!string.IsNullOrEmpty(tableName))
                {
                    tableName = "data_" + tableName;
                    while ((Line = Reader.ReadLine()) != null && !Line.StartsWith(tableName)) ;
                }

                while ((Line = Reader.ReadLine()) != null && !Line.Contains("loop_")) ;

                while (true)
                {
                    Line = Reader.ReadLine();

                    if (Line == null)
                        break;
                    if (Line.Length == 0)
                        continue;
                    if (Line[0] != '_')
                        break;

                    string[] Parts = Line.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                    string ColumnName = Parts[0].Substring(1);
                    int ColumnIndex = Parts.Length > 1 ? int.Parse(Parts[1].Substring(1)) - 1 : NameMapping.Count;
                    NameMapping.Add(ColumnName, ColumnIndex);
                }

                do
                {
                    if (Line == null)
                        break;

                    string[] Parts = Line.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                    if (Parts.Length == NameMapping.Count)
                        Rows.Add(new List<string>(Parts));
                    else
                        break;

                    if (nrows > 0 && Rows.Count >= nrows)
                        break;
                } while ((Line = Reader.ReadLine()) != null);
            }
        }

        public Star(string[] columnNames)
        {
            foreach (string name in columnNames)
                NameMapping.Add(name, NameMapping.Count);
        }

        public Star(Star[] tables)
        {
            List<string> Common = new List<string>(tables[0].GetColumnNames());

            foreach (var table in tables)
                Common.RemoveAll(c => !table.HasColumn(c));

            foreach (string name in Common)
                NameMapping.Add(name, NameMapping.Count);

            foreach (var table in tables)
            {
                int[] ColumnIndices = Common.Select(c => table.GetColumnID(c)).ToArray();

                for (int r = 0; r < table.RowCount; r++)
                {
                    List<string> Row = new List<string>(Common.Count);
                    for (int c = 0; c < ColumnIndices.Length; c++)
                        Row.Add(table.GetRowValue(r, ColumnIndices[c]));

                    AddRow(Row);
                }
            }
        }

        public Star(float[] values, string nameColumn1)
        {
            AddColumn(nameColumn1, values.Select(v => v.ToString(CultureInfo.InvariantCulture)).ToArray());
        }

        public Star(float2[] values, string nameColumn1, string nameColumn2)
        {
            AddColumn(nameColumn1, values.Select(v => v.X.ToString(CultureInfo.InvariantCulture)).ToArray());
            AddColumn(nameColumn2, values.Select(v => v.Y.ToString(CultureInfo.InvariantCulture)).ToArray());
        }

        public Star(float3[] values, string nameColumn1, string nameColumn2, string nameColumn3)
        {
            AddColumn(nameColumn1, values.Select(v => v.X.ToString(CultureInfo.InvariantCulture)).ToArray());
            AddColumn(nameColumn2, values.Select(v => v.Y.ToString(CultureInfo.InvariantCulture)).ToArray());
            AddColumn(nameColumn3, values.Select(v => v.Z.ToString(CultureInfo.InvariantCulture)).ToArray());
        }

        public Star(float4[] values, string nameColumn1, string nameColumn2, string nameColumn3, string nameColumn4)
        {
            AddColumn(nameColumn1, values.Select(v => v.X.ToString(CultureInfo.InvariantCulture)).ToArray());
            AddColumn(nameColumn2, values.Select(v => v.Y.ToString(CultureInfo.InvariantCulture)).ToArray());
            AddColumn(nameColumn3, values.Select(v => v.Z.ToString(CultureInfo.InvariantCulture)).ToArray());
            AddColumn(nameColumn4, values.Select(v => v.W.ToString(CultureInfo.InvariantCulture)).ToArray());
        }

        public Star(string[][] columnValues, params string[] columnNames)
        {
            if (columnValues.Length != columnNames.Length)
                throw new DimensionMismatchException();

            for (int i = 0; i < columnNames.Length; i++)
                AddColumn(columnNames[i], columnValues[i]);
        }

        public static Dictionary<string, Star> FromMultitable(string path, IEnumerable<string> names)
        {
            return names.Where(name => ContainsTable(path, name)).ToDictionary(name => name, name => new Star(path, name));
        }

        public static bool ContainsTable(string path, string name)
        {
            bool Found = false;
            name = "data_" + name;

            using (TextReader Reader = File.OpenText(path))
            {
                string Line;
                while ((Line = Reader.ReadLine()) != null)
                    if (Line.StartsWith(name))
                    {
                        Found = true;
                        break;
                    }
            }

            return Found;
        }

        public static void SaveMultitable(string path, Dictionary<string, Star> tables)
        {
            bool WrittenFirst = false;
            foreach (var pair in tables)
            {
                pair.Value.Save(path, pair.Key, WrittenFirst);
                WrittenFirst = true;
            }
        }

        public void Save(string path, string name = "", bool append = false)
        {
            using (TextWriter Writer = append ? File.AppendText(path) : File.CreateText(path))
            {
                if (append)
                    Writer.WriteLine("\n\n");

                Writer.WriteLine("");
                Writer.WriteLine("data_" + name);
                Writer.WriteLine("");
                Writer.WriteLine("loop_");

                foreach (var pair in NameMapping)
                    Writer.WriteLine($"_{pair.Key} #{pair.Value + 1}");

                int[] ColumnWidths = new int[ColumnCount];
                foreach (var row in Rows)
                    for (int i = 0; i < row.Count; i++)
                        ColumnWidths[i] = Math.Max(ColumnWidths[i], row[i].Length);
                int RowLength = ColumnWidths.Select(v => v + 2).Sum();

                foreach (var row in Rows)
                {
                    StringBuilder RowBuilder = new StringBuilder(RowLength);
                    for (int i = 0; i < row.Count; i++)
                    {
                        RowBuilder.Append(' ', 2 + ColumnWidths[i] - row[i].Length);
                        RowBuilder.Append(row[i]);
                    }
                    Writer.WriteLine(RowBuilder.ToString());
                }
            }
        }

        public static int CountLines(string path)
        {
            int Result = 0;

            using (TextReader Reader = new StreamReader(File.OpenRead(path)))
            {
                string Line;

                while ((Line = Reader.ReadLine()) != null && !Line.Contains("loop_")) ;

                while (true)
                {
                    Line = Reader.ReadLine();

                    if (Line == null)
                        break;
                    if (Line.Length == 0)
                        continue;
                    if (Line[0] != '_')
                        break;
                }

                do
                {
                    if (Line == null)
                        break;

                    if (Line.Length > 3)
                        Result++;

                } while ((Line = Reader.ReadLine()) != null);
            }

            return Result;
        }

        public string[] GetColumn(string name)
        {
            if (!NameMapping.ContainsKey(name))
                return null;

            int Index = NameMapping[name];
            string[] Column = new string[Rows.Count];
            for (int i = 0; i < Rows.Count; i++)
                Column[i] = Rows[i][Index];

            return Column;
        }

        public string[] GetColumn(int id)
        {
            string[] Column = new string[Rows.Count];
            for (int i = 0; i < Rows.Count; i++)
                Column[i] = Rows[i][id];

            return Column;
        }

        public void SetColumn(string name, string[] values)
        {
            int Index = NameMapping[name];
            for (int i = 0; i < Rows.Count; i++)
                Rows[i][Index] = values[i];
        }

        public int GetColumnID(string name)
        {
            if (NameMapping.ContainsKey(name))
                return NameMapping[name];
            else
                return -1;
        }

        public List<List<string>> GetAllRows()
        {
            return Rows;
        }

        public string GetRowValue(int row, string column)
        {
            if (!NameMapping.ContainsKey(column))
                throw new Exception("Column does not exist.");
            if (row < 0 || row >= Rows.Count)
                throw new Exception("Row does not exist.");

            return GetRowValue(row, NameMapping[column]);
        }

        public string GetRowValue(int row, int column)
        {
            return Rows[row][column];
        }

        public float GetRowValueFloat(int row, string column)
        {
            if (!NameMapping.ContainsKey(column))
                throw new Exception("Column does not exist.");
            if (row < 0 || row >= Rows.Count)
                throw new Exception("Row does not exist.");

            return GetRowValueFloat(row, NameMapping[column]);
        }

        public float GetRowValueFloat(int row, int column)
        {
            return float.Parse(Rows[row][column], CultureInfo.InvariantCulture);
        }

        public int GetRowValueInt(int row, string column)
        {
            if (!NameMapping.ContainsKey(column))
                throw new Exception("Column does not exist.");
            if (row < 0 || row >= Rows.Count)
                throw new Exception("Row does not exist.");

            return GetRowValueInt(row, NameMapping[column]);
        }

        public int GetRowValueInt(int row, int column)
        {
            return int.Parse(Rows[row][column]);
        }

        public void SetRowValue(int row, string column, string value)
        {
            Rows[row][NameMapping[column]] = value;
        }

        public void SetRowValue(int row, int column, string value)
        {
            Rows[row][column] = value;
        }

        public void SetRowValue(int row, string column, float value)
        {
            Rows[row][NameMapping[column]] = value.ToString(CultureInfo.InvariantCulture);
        }

        public void SetRowValue(int row, string column, int value)
        {
            Rows[row][NameMapping[column]] = value.ToString();
        }

        public void ModifyAllValuesInColumn(string columnName, Func<string, string> f)
        {
            int ColumnID = GetColumnID(columnName);
            for (int r = 0; r < Rows.Count; r++)
                Rows[r][ColumnID] = f(Rows[r][ColumnID]);
        }

        public void ModifyAllValuesInColumn(string columnName, Func<string, int, string> f)
        {
            int ColumnID = GetColumnID(columnName);
            for (int r = 0; r < Rows.Count; r++)
                Rows[r][ColumnID] = f(Rows[r][ColumnID], r);
        }

        public bool HasColumn(string name)
        {
            return NameMapping.ContainsKey(name);
        }

        public void AddColumn(string name, string[] values)
        {
            int NewIndex = NameMapping.Count > 0 ? NameMapping.Select((v, k) => k).Max() + 1 : 0;
            NameMapping.Add(name, NewIndex);

            if (Rows.Count == 0)
                Rows = Helper.ArrayOfFunction(i => new List<string>(), values.Length).ToList();

            if (Rows.Count != values.Length)
                throw new DimensionMismatchException();

            for (int i = 0; i < Rows.Count; i++)
                Rows[i].Insert(NewIndex, values[i]);
        }

        public void AddColumn(string name, string defaultValue = "")
        {
            string[] EmptyValues = Helper.ArrayOfConstant(defaultValue, RowCount);

            AddColumn(name, EmptyValues);
        }

        public void RemoveColumn(string name)
        {
            int Index = NameMapping[name];
            foreach (List<string> row in Rows)
                row.RemoveAt(Index);

            NameMapping.Remove(name);
            var BiggerNames = NameMapping.Where(vk => vk.Value > Index).Select(vk => vk.Key).ToArray();
            foreach (var biggerName in BiggerNames)
                NameMapping[biggerName] = NameMapping[biggerName] - 1;

            var KeyValuePairs = NameMapping.Select(vk => vk).ToList();
            KeyValuePairs.Sort((vk1, vk2) => vk1.Value.CompareTo(vk2.Value));
            NameMapping = new Dictionary<string, int>();
            foreach (var keyValuePair in KeyValuePairs)
                NameMapping.Add(keyValuePair.Key, keyValuePair.Value);
        }

        public string[] GetColumnNames()
        {
            return NameMapping.Select(pair => pair.Key).ToArray();
        }

        public List<string> GetRow(int index)
        {
            return Rows[index];
        }

        public void AddRow(List<string> row)
        {
            Rows.Add(row);
        }

        public void RemoveRows(int[] indices)
        {
            for (int i = indices.Length - 1; i >= 0; i--)
                Rows.RemoveAt(indices[i]);
        }

        public Star CreateSubset(IEnumerable<int> rows)
        {
            Star Subset = new Star(GetColumnNames());
            foreach (var row in rows)
                Subset.AddRow(new List<string>(Rows[row]));

            return Subset;
        }

        public static float[] LoadFloat(string path, string name1 = null)
        {
            Star TableIn = new Star(path);
            return (name1 == null ? TableIn.GetColumn(0) : TableIn.GetColumn(name1)).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
        }

        public static float2[] LoadFloat2(string path, string name1 = null, string name2 = null)
        {
            Star TableIn = new Star(path);

            float[] Column1 = (name1 == null ? TableIn.GetColumn(0) : TableIn.GetColumn(name1)).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
            float[] Column2 = (name2 == null ? TableIn.GetColumn(1) : TableIn.GetColumn(name2)).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();

            return Helper.Zip(Column1, Column2);
        }

        public static float3[] LoadFloat3(string path, string name1 = null, string name2 = null, string name3 = null)
        {
            Star TableIn = new Star(path);

            float[] Column1 = (name1 == null ? TableIn.GetColumn(0) : TableIn.GetColumn(name1)).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
            float[] Column2 = (name2 == null ? TableIn.GetColumn(1) : TableIn.GetColumn(name2)).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
            float[] Column3 = (name3 == null ? TableIn.GetColumn(2) : TableIn.GetColumn(name3)).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();

            return Helper.Zip(Column1, Column2, Column3);
        }

        public static float4[] LoadFloat4(string path)
        {
            Star TableIn = new Star(path);
            float[] Column1 = TableIn.GetColumn(0).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
            float[] Column2 = TableIn.GetColumn(1).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
            float[] Column3 = TableIn.GetColumn(2).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
            float[] Column4 = TableIn.GetColumn(3).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();

            return Helper.Zip(Column1, Column2, Column3, Column4);
        }

        public static float[][] LoadFloatN(string path, int n = -1)
        {
            Star TableIn = new Star(path);
            if (n < 0)
                n = TableIn.ColumnCount;

            float[][] Result = new float[TableIn.RowCount][];
            for (int r = 0; r < TableIn.RowCount; r++)
                Result[r] = TableIn.GetRow(r).Take(n).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();

            return Result;
        }
    }
}