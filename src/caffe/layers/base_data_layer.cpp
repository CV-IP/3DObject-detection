#include <string>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/net.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
BaseDataLayer<Dtype>::BaseDataLayer(const LayerParameter& param)
    : Layer<Dtype>(param),
      transform_param_(param.transform_param()) {
}

template <typename Dtype>
void BaseDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (top.size() == 1) {
    output_labels_ = false;
  } else {
    output_labels_ = true;
  }
  // The subclasses should setup the size of bottom and top
  DataLayerSetUp(bottom, top);
  data_transformer_.reset(
      new DataTransformer<Dtype>(transform_param_, this->phase_));
  data_transformer_->InitRand();
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);
  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  this->prefetch_data_.mutable_cpu_data();
  if (this->output_labels_) {
    this->prefetch_label_.mutable_cpu_data();
  }
  DLOG(INFO) << "Initializing prefetch";
  this->CreatePrefetchThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::CreatePrefetchThread() {
  this->data_transformer_->InitRand();
  CHECK(StartInternalThread()) << "Thread execution failed";
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::JoinPrefetchThread() {
  CHECK(WaitForInternalThreadToExit()) << "Thread joining failed";
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  JoinPrefetchThread();
  DLOG(INFO) << "Thread joined";
  // Reshape to loaded data.
  top[0]->Reshape(this->prefetch_data_.num(), this->prefetch_data_.channels(),
      this->prefetch_data_.height(), this->prefetch_data_.width());
  // Copy the data
  caffe_copy(prefetch_data_.count(), prefetch_data_.cpu_data(),
             top[0]->mutable_cpu_data());
  DLOG(INFO) << "Prefetch copied";
  if (this->output_labels_) {
    caffe_copy(prefetch_label_.count(), prefetch_label_.cpu_data(),
               top[1]->mutable_cpu_data());
  }
  // Start a new prefetch thread
  DLOG(INFO) << "CreatePrefetchThread";
  CreatePrefetchThread();
}

template <typename Dtype>
void Scene3DBasePrefetchingDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);
  if (top.size() > 3) {
    output_bb3d_diff_ = true;
  } else {
    output_bb3d_diff_ = false;
  }
  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  this->prefetch_data_.mutable_cpu_data();
  if (this->output_labels_) {
    this->prefetch_label_.mutable_cpu_data();
  }

  this->prefetch_bb2d_proj_.mutable_cpu_data();
  if (this->output_bb3d_diff_) {
    this->prefetch_bb3d_diff_.mutable_cpu_data();
  }

  DLOG(INFO) << "Initializing prefetch";
  BasePrefetchingDataLayer<Dtype>::CreatePrefetchThread();
  DLOG(INFO) << "Prefetch initialized.";
}


template <typename Dtype>
void Scene3DBasePrefetchingDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  BasePrefetchingDataLayer<Dtype>::JoinPrefetchThread();
  DLOG(INFO) << "Thread joined";
  // Reshape to loaded data.
  top[0]->Reshape(this->prefetch_data_.shape(0), this->prefetch_data_.shape(1),
      this->prefetch_data_.shape(2), this->prefetch_data_.shape(3), this->prefetch_data_.shape(4));
  // Copy the data
  CUDA_CHECK(cudaMemcpy(top[0]->mutable_cpu_data(), this->prefetch_data_.gpu_data(), 
    this->prefetch_data_.count() * sizeof(Dtype), cudaMemcpyDefault));

  DLOG(INFO) << "Prefetch copied";
  if (this->output_labels_) {
    caffe_copy(this->prefetch_label_.count(), this->prefetch_label_.cpu_data(),
               top[1]->mutable_cpu_data());
  }

  caffe_copy(this->prefetch_bb2d_proj_.count(), this->prefetch_bb2d_proj_.cpu_data(),
    top[2]->mutable_cpu_data());
  caffe_copy(this->prefetch_attention_bb_.count(), this->prefetch_attention_bb_.cpu_data(),
    top[3]->mutable_cpu_data());

  if (this->output_bb3d_diff_)
  {
    caffe_copy(this->prefetch_bb3d_diff_.count(), this->prefetch_bb3d_diff_.cpu_data(),
      top[4]->mutable_cpu_data());
    caffe_copy(this->prefetch_bb3d_param_.count(), this->prefetch_bb3d_param_.cpu_data(),
      top[5]->mutable_cpu_data());
  }

  // Start a new prefetch thread
  DLOG(INFO) << "CreatePrefetchThread";
  BasePrefetchingDataLayer<Dtype>::CreatePrefetchThread();
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(BasePrefetchingDataLayer, Forward);
STUB_GPU_FORWARD(Scene3DBasePrefetchingDataLayer, Forward);
#endif

INSTANTIATE_CLASS(BaseDataLayer);
INSTANTIATE_CLASS(BasePrefetchingDataLayer);
INSTANTIATE_CLASS(Scene3DBasePrefetchingDataLayer);

}  // namespace caffe
