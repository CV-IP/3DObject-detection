#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"

namespace caffe {

template <typename Dtype>
void ReshapeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BlobShape blobShape = this->layer_param_.reshape_param().shape();
  vector<int> shape;
  int count = bottom[0]->count();
  for (int i = 0; i < blobShape.dim_size(); i++)
  {
  	if (blobShape.dim(i) == 0)
  	{
  		shape.push_back(bottom[0]->shape(i));
  		count /= bottom[0]->shape(i);
  	}
  	else if (blobShape.dim(i) == -1)
  	{
  		shape.push_back(count);
  		break;
  	}
  	else
  	{
  		shape.push_back(blobShape.dim(i));
  		count /= blobShape.dim(i);
  	}
  }

  top[0]->Reshape(shape);
  CHECK_EQ(top[0]->count(), bottom[0]->count())
     << "new shape must have the same count as input";
  top[0]->ShareData(*bottom[0]);
  top[0]->ShareDiff(*bottom[0]);
}

INSTANTIATE_CLASS(ReshapeLayer);
REGISTER_LAYER_CLASS(Reshape);

}  // namespace caffe
