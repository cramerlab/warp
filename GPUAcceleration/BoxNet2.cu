#include "Functions.h"
#include "../../gtom/include/DeviceFunctions.cuh"
using namespace gtom;

__global__ void BoxNet2AugmentKernel(cudaTex t_imagemic,
                                    float* d_inputlabel,
                                    float* d_inputuncertain,
                                    int2 dimsinput,
                                    float* d_outputmic,
                                    float3* d_outputlabel,
                                    float* d_outputweights,
                                    uint dimoutput,
                                    float3 labelweights,
                                    float2* d_offsets,
                                    glm::mat2* d_transforms);

void BoxNet2Augment(float* d_inputmic, 
                    float* d_inputlabel, 
                    float* d_inputuncertain, 
                    int2 dimsinput, 
                    float* d_outputmic, 
                    float3* d_outputlabel, 
                    float* d_outputweights, 
                    int2 dimsoutput, 
                    float3 labelweights,
                    float2* h_offsets, 
                    float* h_rotations, 
                    float3* h_scales, 
                    float noisestddev, 
                    int seed,
                    uint batch)
{
    cudaArray_t a_inputmic;
    cudaTex t_inputmic;

    {
        d_BindTextureToArray(d_inputmic, a_inputmic, t_inputmic, toInt2(dimsinput.x, dimsinput.y), cudaFilterModePoint, false);
    }

    glm::mat2* h_transforms = (glm::mat2*)malloc(batch * sizeof(glm::mat2));
    for (int i = 0; i < batch; i++)
        h_transforms[i] = Matrix2Rotation(h_rotations[i]) * Matrix2Rotation(h_scales[i].z) * Matrix2Scale(tfloat2(1.0f / h_scales[i].x, 1.0f / h_scales[i].y)) * Matrix2Rotation(-h_scales[i].z);

    glm::mat2* d_transforms = (glm::mat2*)CudaMallocFromHostArray(h_transforms, batch * sizeof(glm::mat2));
    free(h_transforms);

    float2* d_offsets = (float2*)CudaMallocFromHostArray(h_offsets, batch * sizeof(float2));

    dim3 grid = dim3(tmin(32768, (Elements2(dimsoutput) + 127) / 128), batch, 1);
    BoxNet2AugmentKernel << <grid, 128 >> > (t_inputmic, d_inputlabel, d_inputuncertain, dimsinput, d_outputmic, d_outputlabel, d_outputweights, dimsoutput.x, labelweights, d_offsets, d_transforms);

    cudaFree(d_offsets);
    cudaFree(d_transforms);

    {
        cudaDestroyTextureObject(t_inputmic);
        cudaFreeArray(a_inputmic);
    }

    float* d_noise = CudaMallocRandomFilled(Elements2(dimsoutput) * batch, 0, noisestddev, seed);
    d_AddVector(d_outputmic, d_noise, d_outputmic, Elements2(dimsoutput) * batch);

    cudaFree(d_noise);

    d_NormMonolithic(d_outputmic, d_outputmic, Elements2(dimsoutput), T_NORM_MEAN01STD, batch);
}

__global__ void BoxNet2AugmentKernel(cudaTex t_inputmic, 
                                    float* d_inputlabel, 
                                    float* d_inputuncertain, 
                                    int2 dimsinput, 
                                    float* d_outputmic, 
                                    float3* d_outputlabel, 
                                    float* d_outputweights, 
                                    uint dimoutput, 
                                    float3 labelweights,
                                    float2* d_offsets, 
                                    glm::mat2* d_transforms)
{
    d_outputmic += dimoutput * dimoutput * blockIdx.y;
    d_outputlabel += dimoutput * dimoutput * blockIdx.y;
    d_outputweights += dimoutput * dimoutput * blockIdx.y;

    int outputcenter = dimoutput / 2;

    glm::mat2 transform = d_transforms[blockIdx.y];
    float2 offset = d_offsets[blockIdx.y];

    for (uint id = blockIdx.x * blockDim.x + threadIdx.x; id < dimoutput * dimoutput; id += gridDim.x * blockDim.x)
    {
        uint idx = id % dimoutput;
        uint idy = id / dimoutput;

        int posx = (int)idx - outputcenter;
        int posy = (int)idy - outputcenter;

        glm::vec2 pos = transform * glm::vec2(posx, posy);

        pos.x += outputcenter + offset.x;
        pos.y += outputcenter + offset.y;

        /*if (pos.x > dimsinput.x - 1)
            pos.x = dimsinput.x * 2 - 2 - pos.x;
        if (pos.y > dimsinput.y - 1)
            pos.y = dimsinput.y * 2 - 2 - pos.y;
        pos.x = abs(pos.x);
        pos.y = abs(pos.y);*/
        
        float val = 0;
        float3 label = make_float3(1, 0, 0);
        float weight = labelweights.x;

        if (pos.x > 0 && pos.y > 0 && pos.x < dimsinput.x - 1 && pos.y < dimsinput.y - 1)
        {
            for (int y = -4; y <= 4; y++)
            {
                float yy = floor(pos.y) + y;
                float sincy = sinc(pos.y - yy);
                float yy2 = pos.y - yy;
                yy2 *= yy2;
                yy += 0.5f;

                for (int x = -4; x <= 4; x++)
                {
                    float xx = floor(pos.x) + x;
                    float sincx = sinc(pos.x - xx);
                    float xx2 = pos.x - xx;
                    xx2 *= xx2;
                    float r2 = xx2 + yy2;

                    if (r2 > 16)
                        continue;

                    float hanning = 1.0f + cos(PI * sqrt(r2) / 4);

                    val += tex2D<float>(t_inputmic, xx + 0.5f, yy) * sincy * sincx * hanning;
                }
            }

            int labelcompressed = (int)d_inputlabel[(int)(pos.y + 0.5f) * dimsinput.x + (int)(pos.x + 0.5f)];
            label = make_float3(labelcompressed == 0 ? 1 : 0,
                labelcompressed == 1 ? 1 : 0,
                labelcompressed == 2 ? 1 : 0);

            float uncertaincompressed = d_inputuncertain[(int)(pos.y + 0.5f) * dimsinput.x + (int)(pos.x + 0.5f)];
            weight = uncertaincompressed * (labelcompressed == 2 ? labelweights.z : (labelcompressed == 1 ? labelweights.y : labelweights.x));
        }

        d_outputmic[id] = val * 0.5f;
        d_outputlabel[id] = label;
        d_outputweights[id] = weight;
    }
}