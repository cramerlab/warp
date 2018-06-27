#include "Functions.h"
using namespace gtom;

float CubicInterpShort(float2* data, float x, int n);
template<int N> float CubicInterp(float2* data, float x);

__declspec(dllexport) void __stdcall CubicInterpOnGrid(int3 dimensions, float* values, float3 spacing, int3 valueGrid, float3 step, float3 offset, float* output)
{
	///#pragma omp parallel for
	for (int valueZ = 0; valueZ < valueGrid.z; valueZ++)
		for (int valueY = 0; valueY < valueGrid.y; valueY++)
			for (int valueX = 0; valueX < valueGrid.x; valueX++)
			{
				float3 coords = make_float3(valueX * step.x + offset.x, valueY * step.y + offset.y, valueZ * step.z + offset.z);
				
				{
					coords = make_float3(coords.x / spacing.x, coords.y / spacing.y, coords.z / spacing.z);  // from [0, 1] to [0, dim - 1]

					float3 coord_grid = coords;
					float3 index = make_float3(floor(coord_grid.x), floor(coord_grid.y), floor(coord_grid.z));

					float result = 0.0f;

					int MinX = tmax(0, (int)index.x - 1), MaxX = tmin((int)index.x + 2, dimensions.x - 1);
					int MinY = tmax(0, (int)index.y - 1), MaxY = tmin((int)index.y + 2, dimensions.y - 1);
					int MinZ = tmax(0, (int)index.z - 1), MaxZ = tmin((int)index.z + 2, dimensions.z - 1);

					int nz = MaxZ - MinZ + 1;
					int ny = MaxY - MinY + 1;
					int nx = MaxX - MinX + 1;

					float InterpX[16];
					for (int z = MinZ; z <= MaxZ; z++)
					{
						for (int y = MinY; y <= MaxY; y++)
						{
							float2 Points[4];
							if (nx == 1)
								InterpX[(z - MinZ) * ny + y - MinY] = values[(z * dimensions.y + y) * 1];
							else
							{
								for (int x = MinX; x <= MaxX; x++)
									Points[x - MinX] = make_float2(x, values[(z * dimensions.y + y) * dimensions.x + x]);

								InterpX[(z - MinZ) * ny + y - MinY] = CubicInterpShort(Points, coords.x, nx);
							}
						}
					}

					float InterpXY[4];
					for (int z = MinZ; z <= MaxZ; z++)
					{
						float2 Points[4];
						if (ny == 1)
							InterpXY[z - MinZ] = InterpX[(z - MinZ) * ny];
						else
						{
							for (int y = MinY; y <= MaxY; y++)
								Points[y - MinY] = make_float2(y, InterpX[(z - MinZ) * ny + y - MinY]);

							InterpXY[z - MinZ] = CubicInterpShort(Points, coords.y, ny);
						}
					}

					{
						float2 Points[4];
						if (nz == 1)
							result = InterpXY[0];
						else
						{
							for (int z = MinZ; z <= MaxZ; z++)
								Points[z - MinZ] = make_float2(z, InterpXY[z - MinZ]);

							result = CubicInterpShort(Points, coords.z, nz);
						}
					}

					output[(valueZ * valueGrid.y + valueY) * valueGrid.x + valueX] = result;
				}
			}
}

__declspec(dllexport) void __stdcall CubicInterpIrregular(int3 dimensions, float* values, float3* positions, int npositions, float3 spacing, float3 samplingmargin, float3 samplingmarginscale, float* output)
{
///#pragma omp parallel for
    for (int position = 0; position < npositions; position++)
    {
        float3 coords = positions[position];

        {
            coords = make_float3(coords.x * samplingmarginscale.x + samplingmargin.x, coords.y * samplingmarginscale.y + samplingmargin.y, coords.z * samplingmarginscale.z + samplingmargin.z);
            coords = make_float3(coords.x / spacing.x, coords.y / spacing.y, coords.z / spacing.z);  // from [0, 1] to [0, dim - 1]

            float3 coord_grid = coords;
            float3 index = make_float3(floor(coord_grid.x), floor(coord_grid.y), floor(coord_grid.z));

            float result = 0.0f;

            int MinX = tmax(0, (int)index.x - 1), MaxX = tmin((int)index.x + 2, dimensions.x - 1);
            int MinY = tmax(0, (int)index.y - 1), MaxY = tmin((int)index.y + 2, dimensions.y - 1);
            int MinZ = tmax(0, (int)index.z - 1), MaxZ = tmin((int)index.z + 2, dimensions.z - 1);

            int nz = MaxZ - MinZ + 1;
            int ny = MaxY - MinY + 1;
            int nx = MaxX - MinX + 1;

            float InterpX[16];
            for (int z = MinZ; z <= MaxZ; z++)
            {
                for (int y = MinY; y <= MaxY; y++)
                {
                    float2 Points[4];
                    if (nx == 1)
                        InterpX[(z - MinZ) * ny + y - MinY] = values[(z * dimensions.y + y) * 1];
                    else
                    {
                        for (int x = MinX; x <= MaxX; x++)
                            Points[x - MinX] = make_float2(x, values[(z * dimensions.y + y) * dimensions.x + x]);

                        InterpX[(z - MinZ) * ny + y - MinY] = CubicInterpShort(Points, coords.x, nx);
                    }
                }
            }

            float InterpXY[4];
            for (int z = MinZ; z <= MaxZ; z++)
            {
                float2 Points[4];
                if (ny == 1)
                    InterpXY[z - MinZ] = InterpX[(z - MinZ) * ny];
                else
                {
                    for (int y = MinY; y <= MaxY; y++)
                        Points[y - MinY] = make_float2(y, InterpX[(z - MinZ) * ny + y - MinY]);

                    InterpXY[z - MinZ] = CubicInterpShort(Points, coords.y, ny);
                }
            }

            {
                float2 Points[4];
                if (nz == 1)
                    result = InterpXY[0];
                else
                {
                    for (int z = MinZ; z <= MaxZ; z++)
                        Points[z - MinZ] = make_float2(z, InterpXY[z - MinZ]);

                    result = CubicInterpShort(Points, coords.z, nz);
                }
            }

            output[position] = result;
        }
    }
}

float CubicInterpShort(float2* data, float x, int n)
{
	if (n == 4)
		return CubicInterp<4>(data, x);
	if (n == 3)
		return CubicInterp<3>(data, x);

	return CubicInterp<2>(data, x);
}

template<int N> float CubicInterp(float2* data, float x)
{
	float2* Data;
	float Breaks[N];
	float4 Coefficients[N - 1];
	
	Data = data;
	for (int i = 0; i < N; i++)
		Breaks[i] = data[i].x;

	float h[N - 1];
	for (int i = 0; i < N - 1; i++)
		h[i] = data[i + 1].x - data[i].x;

	float del[N - 1];
	for (int i = 0; i < N - 1; i++)
		del[i] = (data[i + 1].y - data[i].y) / h[i];

	float slopes[N] = { 0 };
	{
		if (N == 2)
			slopes[0] = slopes[1] = del[0];   // Do only linear
		else
		{
			for (int k = 0; k < N - 2; k++)
			{
				if (del[k] * del[k + 1] <= 0.0f)
					continue;

				float hs = h[k] + h[k + 1];
				float w1 = (h[k] + hs) / (3.0f * hs);
				float w2 = (hs + h[k + 1]) / (3.0f * hs);
				float dmax = tmax(abs(del[k]), abs(del[k + 1]));
				float dmin = tmin(abs(del[k]), abs(del[k + 1]));
				slopes[k + 1] = dmin / (w1 * (del[k] / dmax) + w2 * (del[k + 1] / dmax));
			}

			slopes[0] = ((2.0f * h[0] + h[1]) * del[0] - h[0] * del[1]) / (h[0] + h[1]);
			if (sgn(slopes[0]) != sgn(del[0]))
				slopes[0] = 0;
			else if (sgn(del[0]) != sgn(del[1]) && abs(slopes[0]) > abs(3.0f * del[0]))
				slopes[0] = 3.0f * del[0];

			int n = N - 1;
			slopes[n] = ((2 * h[n - 1] + h[n - 2]) * del[n - 1] - h[n - 1] * del[n - 2]) / (h[n - 1] + h[n - 2]);
			if (sgn(slopes[n]) != sgn(del[n - 1]))
				slopes[n] = 0;
			else if (sgn(del[n - 1]) != sgn(del[n - 2]) && abs(slopes[n]) > abs(3.0f * del[n - 1]))
				slopes[n] = 3.0f * del[n - 1];
		}
	}

	float dzzdx[N - 1];
	for (int i = 0; i < N - 1; i++)
		dzzdx[i] = (del[i] - slopes[i]) / h[i];

	float dzdxdx[N - 1];
	for (int i = 0; i < N - 1; i++)
		dzdxdx[i] = (slopes[i + 1] - del[i]) / h[i];

	for (int i = 0; i < N - 1; i++)
		Coefficients[i] = make_float4((dzdxdx[i] - dzzdx[i]) / h[i],
									  2.0f * dzzdx[i] - dzdxdx[i],
									  slopes[i],
									  data[i].y);

	// Now interpolate
	float* b = Breaks;
	float4* c = Coefficients;

	int index = 0;

	if (x < b[1])
		index = 0;
	else if (x >= b[N - 2])
		index = N - 2;
	else
		for (int j = 2; j < N - 1; j++)
			if (x < b[j])
			{
				index = j - 1;
				break;
			}

	float xs = x - b[index];

	float v = c[index].x;
	v = xs * v + c[index].y;
	v = xs * v + c[index].z;
	v = xs * v + c[index].w;

	float y = v;

	return y;
}