#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/matrix_multiply_layer.hpp"

namespace caffe {
template <typename Dtype>	
void MatrixMultiplyLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
	CHECK_EQ(bottom.size(), 2) << "There must be exact 2 bottom blobs";
	top[0]->Reshape(bottom[1]->num(), bottom[1]->channels(), 1, 1);
}

template <typename Dtype>
void MatrixMultiplyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
	const Dtype* bottom_data1 = bottom[0]->cpu_data();
	const Dtype* bottom_data2 = bottom[1]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();

	for (int n = 0; n < bottom[1]->num(); n++)
	{
		for (int c = 0; c < bottom[1]->channels(); c++)
		{
			Dtype sum = Dtype(0);
			for (int i = 0; i < bottom[1]->height() * bottom[1]->width(); i++)
			{
				sum += bottom_data1[i] * bottom_data2[i];
			}
			top_data[c] = sum;
			bottom_data2 += bottom[1]->offset(0, 1);
		}
		top_data += top[0]->offset(1);
		bottom_data1 += bottom[0]->offset(1);
	}
}

template <typename Dtype>
void MatrixMultiplyLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	const Dtype* top_diff = top[0]->cpu_diff();
	const Dtype* bottom_data = bottom[1]->cpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

	caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);

	for (int n = 0; n < bottom[0]->num(); n++)
	{
		for (int c = 0; c < top[0]->channels(); c++)
		{
			for (int i = 0; i < bottom[0]->height() * bottom[0]->width(); i++)
			{
				bottom_diff[i] += bottom_data[i] * top_diff[c];
			}

			bottom_data += bottom[1]->offset(0, 1);
		}	

		top_diff += top[0]->offset(1);
		bottom_diff += bottom[0]->offset(1);
	}
}

#ifdef CPU_ONLY
STUB_GPU(MatrixMultiplyLayer);
#endif

INSTANTIATE_CLASS(MatrixMultiplyLayer);
REGISTER_LAYER_CLASS(MatrixMultiply);

}