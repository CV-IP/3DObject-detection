#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/matrix_multiply_layer.hpp"

namespace caffe{

template <typename Dtype>
__global__ void MatrixMultiplyForward(const int nthreads, const Dtype* bottom_data1, const Dtype* bottom_data2,
		   const int num, const int channels, const int height, const int width, Dtype* top_data)
{
	CUDA_KERNEL_LOOP(index, nthreads) {
		int n = index / channels;
		int c = index % channels;

		bottom_data2 += (n * channels + c) * height * width;
		bottom_data1 += n * height * width;

		Dtype sum = Dtype(0);
		for (int i = 0; i < height * width; i++)
		{
			sum += bottom_data1[i] * bottom_data2[i];
		}
		top_data[index] = sum;
	}
}

template <typename Dtype>
void MatrixMultiplyLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data1 = bottom[0]->gpu_data();
	const Dtype* bottom_data2 = bottom[1]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();

	int count = top[0]->count();
	MatrixMultiplyForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, bottom_data1,
		bottom_data2, bottom[1]->num(), bottom[1]->channels(), bottom[1]->height(), bottom[1]->width(), top_data);
}

template <typename Dtype>
__global__ void MatrixMultiplyBackward(const int nthreads, const Dtype* top_diff, const Dtype* bottom_data,
		   const int num, const int channels, const int height, const int width, Dtype* bottom_diff)
{
	CUDA_KERNEL_LOOP(index, nthreads) {
		int n = index / channels;
		int c = index % channels;

		bottom_data += (n * channels + c) * height * width;
		bottom_diff += n * height * width;
		top_diff += n * channels;

		for (int i = 0; i < height * width; i++)
		{
			bottom_diff[i] += bottom_data[i] * top_diff[c];
		}
	}
}

template <typename Dtype>
void MatrixMultiplyLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	const Dtype* top_diff = top[0]->gpu_diff();
	const Dtype* bottom_data = bottom[1]->gpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	
	cudaMemset(bottom_diff, 0, bottom[0]->count() * sizeof(Dtype));
	//caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
	int count = top[0]->count();
	MatrixMultiplyBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, top_diff,
		bottom_data, bottom[1]->num(), bottom[1]->channels(), bottom[1]->height(), bottom[1]->width(), bottom_diff);	
}

INSTANTIATE_LAYER_GPU_FUNCS(MatrixMultiplyLayer);
}
