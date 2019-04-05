//��ֵ�˲�����Ϊboxfilter


#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "dehaze_kernel.h"
#define TILE_DIM 16
#define BLOCKSIZE 128



__global__ void d_boxfilter_x_global(float *src, float *dst, int width, int height, int r)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int offset = 1;
	int num = (width + 2 * r + 2 * BLOCKSIZE - 1) / (2 * BLOCKSIZE);		//ÿһ���߳̿鱻BLOCKSIZE*2�ָ����num��segment
	int len = num * 2 * BLOCKSIZE;
	int extra = len - r - width;
	float scale = 1.0f / (float)((r << 1) + 1);

	__shared__ float sum[35]; sum[0] = 0;

	extern __shared__ float temp[];

	if (bid < height)
	{
		//�߽����
		for (int i = tid; i < r; i += BLOCKSIZE)
		{
			temp[i] = src[bid*width + 0];								//ǰr�������һ�еĵ�һ��Ԫ��
		}
		//__syncthreads();

		for (int i = tid; i < width; i += BLOCKSIZE)
		{
			temp[r + i] = src[bid * width + i];
		}
		//__syncthreads();

		for (int i = tid; i < extra; i += BLOCKSIZE)
		{
			temp[r + width + i] = src[(bid + 1) * width - 1];			//���extra�������һ�����һ��Ԫ��
		}
		__syncthreads();


		for (int cnt = 0; cnt < num; ++cnt)								//numΪ��������
		{
			int bias = cnt * BLOCKSIZE * 2;
			//__syncthreads();
			//up-sweep phase
			for (int j = BLOCKSIZE; j > 0; j >>= 1)
			{
				if (tid < j)
				{
					int ai = bias + offset * (2 * tid + 1) - 1;
					int bi = bias + offset * (2 * tid + 2) - 1;
					temp[bi] += temp[ai];
				}
				offset *= 2;
				__syncthreads();
			}
			//down-sweep phase		
			if (tid == 0)
			{
				sum[cnt + 1] = temp[(cnt + 1) * BLOCKSIZE * 2 - 1] + sum[cnt]; //֮��ÿ�е�ÿ��segment[i]Ҫ����sum[i]������sum[i]��ʾǰi�����ݿ�������Ԫ�صĺͣ�ÿ�����ݿ��� BLOCKSIZE * 2��С
				temp[(cnt + 1) * BLOCKSIZE * 2 - 1] = 0;
			}
			__syncthreads();
			for (int j = 1; j < (BLOCKSIZE * 2); j *= 2)
			{
				offset >>= 1;
				if (tid < j)
				{
					int ai = bias + offset * (2 * tid + 1) - 1;
					int bi = bias + offset * (2 * tid + 2) - 1;

					float t = temp[ai];
					temp[ai] = temp[bi];
					temp[bi] += t;
				}
				__syncthreads();
			}
		}
		for (int i = tid; i < width; i += BLOCKSIZE)
		{
			float sum_box = temp[i + 2 * r + 1] + sum[(i + 2 * r + 1) / (BLOCKSIZE * 2)] - temp[i] - sum[i / (BLOCKSIZE * 2)];		//sumֻ�ǵ�i + 2 * r + 1֮ǰ������Ԫ��֮�Ͳ�������i + 2 * r + 1��Ԫ��
			dst[bid * width + i] = sum_box * scale;
			//dst[bid * width + i] = temp[i];
			//dst[bid * width + i] = src[bid * width + i];
		}
	}
}


//2018.12.18,����HILLIS����ͼ
//sipenghui
__global__ void d_boxfilter_x_hillis(float *src, float *dst, int width, int height, int r)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int offset = 1;
	int num = (width + 2 * r + 2 * BLOCKSIZE - 1) / (2 * BLOCKSIZE);		//ÿһ���߳̿鱻BLOCKSIZE*2�ָ����num��segment
	int len = num * 2 * BLOCKSIZE;
	int extra = len - r - width;
	float scale = 1.0f / (float)((r << 1) + 1);

	__shared__ float sum[35]; sum[0] = 0;

	extern __shared__ float temp[];

	if (bid < height)
	{
		//�߽����
		for (int i = tid; i < r; i += 2*BLOCKSIZE)						//2019.2.28�޸�CLOCKSIZE��2
		{
			temp[i] = src[bid*width + 0];								//ǰr�������һ�еĵ�һ��Ԫ��
			temp[len*2 + i] = src[bid*width + 0];

		}

		for (int i = tid; i < width; i += 2*BLOCKSIZE)
		{
			temp[r + i] = src[bid * width + i];
			temp[len * 2 + r + i] = src[bid * width + i];

		}

		for (int i = tid; i < extra; i += 2*BLOCKSIZE)
		{
			temp[r + width + i] = src[(bid + 1) * width - 1];			//���extra�������һ�����һ��Ԫ��
			temp[len * 2 + r + width + i] = src[(bid + 1) * width - 1];

		}
		__syncthreads();

		int pout = 0;
		int pin = 1;

		for (int cnt = 0; cnt < num; ++cnt)								//numΪ��������
		{
			int bias = cnt * BLOCKSIZE * 2;

			pout = 0;
			pin = 1;

			for (offset = 1; offset < BLOCKSIZE * 2; offset *= 2)
			{
				pout = 1 - pout;
				pin = 1 - pin;

				//if (tid<2*BLOCKSIZE)
				//	temp[len*2+bias + tid] = temp[bias + tid];
				//__syncthreads();
				//temp[pout*len + bias + tid] = temp[pin*len + bias + tid];		//�汾1

				int ai = pout*len + bias + tid;
				int bi = pin*len + bias + tid;
				if (tid >= offset && tid < 2 * BLOCKSIZE)						//�汾2
					//temp[bias + tid] += temp[bias + tid - offset];	//2019.2.28�˴����ڳ�ͻ
					temp[ai] = temp[bi] + temp[bi - offset];
				else
					temp[ai] = temp[bi];

				__syncthreads();
			}

			if (tid == 2 * BLOCKSIZE-1)
			{
				sum[cnt + 1] = temp[pout*len + bias + tid] + sum[cnt];
			}
			if (tid < 2 * BLOCKSIZE)
				temp[pout*len + bias + tid] = temp[pout*len + bias + tid] - temp[len * 2 + bias + tid];

			__syncthreads();
		}


		for (int i = tid; i < width; i += BLOCKSIZE)
		{
			float sum_box = temp[pout*len + i + 2 * r + 1] + sum[(i + 2 * r + 1) / (BLOCKSIZE * 2)] - temp[pout*len + i] - sum[i / (BLOCKSIZE * 2)];		//sumֻ�ǵ�i + 2 * r + 1֮ǰ������Ԫ��֮�Ͳ�������i + 2 * r + 1��Ԫ��
			dst[bid * width + i] = sum_box * scale;
		}
	}
}


extern "C"
void boxfilter(float *id, float *od, float *d_temp, float *d_temp1, int width, int height, int r)
{

	int num_shared1 = ((width + 2 * r + BLOCKSIZE * 2 - 1) / (BLOCKSIZE * 2)) * 2 * BLOCKSIZE;
	int num_shared2 = ((height + 2 * r + BLOCKSIZE * 2 - 1) / (BLOCKSIZE * 2)) * 2 * BLOCKSIZE;

	dim3 grid1(width / TILE_DIM + 1, height / TILE_DIM + 1);
	dim3 grid2(height / TILE_DIM + 1, width / TILE_DIM + 1);
	dim3 block(TILE_DIM, TILE_DIM);



	//d_boxfilter_x_global << < height, BLOCKSIZE, num_shared1 * sizeof(float) >> > (id, d_temp, width, height, r);
	//transpose << <grid1, block >> >(d_temp1, d_temp, width, height);
	//d_boxfilter_x_global << < width, BLOCKSIZE, num_shared2 * sizeof(float) >> > (d_temp1, od, height, width, r);

	d_boxfilter_x_hillis << < height, BLOCKSIZE * 2, num_shared1 * sizeof(float)*3 >> > (id, d_temp, width, height, r);
	transpose << <grid1, block >> >(d_temp1, d_temp, width, height);
	d_boxfilter_x_hillis << < width, BLOCKSIZE * 2, num_shared2 * sizeof(float)*3 >> > (d_temp1, od, height, width, r);


}
