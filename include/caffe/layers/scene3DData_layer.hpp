#ifndef CAFFE_SCENE3DDATALAYER_HPP
#define CAFFE_SCENE3DDATALAYER_HPP

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/util/dss.hpp"

namespace caffe {

template <typename Dtype>
class Scene3DDataLayer : public Scene3DBasePrefetchingDataLayer<Dtype> {
public:
  explicit Scene3DDataLayer(const LayerParameter& param)
      : Scene3DBasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~Scene3DDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Scene3DData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 4; }

protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void shuffleCategory(int cate);
  virtual void shuffleData(Scene3D* scene, bool neg, bool pos);
  virtual void InternalThreadEntry();
  void compute_attention_area(int scene_id, int obj_id, Dtype* attention_bb_data);

  vector<Scene3D*> scenes;
  vector<int> grid_size;
  vector<int> bb_param_weight;
  vector<int> batch_size;
  vector< vector<ImgObjInd> > imgobj_pos_cates;
  vector<Box3Ddiff>  target_boxes;
  string file_list;
  string data_root;


  int encode_type;
  int context_pad;
  int scale;
  int num_categories;
  int scenes_id_;
};
}


#endif
