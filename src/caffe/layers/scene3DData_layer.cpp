#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/scene3DData_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/dss.hpp"

namespace caffe{

using namespace std;

template <typename Dtype>
Scene3DDataLayer<Dtype>::~Scene3DDataLayer(){
	this->JoinPrefetchThread();
}

template <typename Dtype>
void Scene3DDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
	Scene3DDataParameter scene3d_param = this->layer_param_.scene_3d_data_param();
	data_root = scene3d_param.data_root();
	file_list = scene3d_param.file_list();
	num_categories = scene3d_param.num_categories();

	imgobj_pos_cates.resize(num_categories);

	//get the batch size
	batch_size.push_back(scene3d_param.grid_size(0));
	batch_size.push_back(scene3d_param.grid_size(1));

	//get grid size
	for (int i = 0; i < scene3d_param.grid_size().size(); i++)
	{
		grid_size.push_back(scene3d_param.grid_size(i));
	}

	//initialize the bb_param_weight
	for (int i = 0; i < 6; i++)
	{
		bb_param_weight.push_back(1);
	}

	std::cout << "loading file " << file_list << "\n";
	FILE* fp = fopen(file_list.c_str(), "rb");
	if (fp == NULL) { std::cout << "fail to open file: " << file_list << std::endl; exit(1); }
	size_t file_size;
	while (feof(fp) == 0)
	{
		Scene3D* scene = new Scene3D();

		unsigned int len = 0;
		file_size += fread((void*)(&len), sizeof(unsigned int), 1, fp);    
		if (len == 0) break;
		scene->filename.resize(len);
		if (len > 0) 
			file_size += fread((void*)(scene->filename.data()), sizeof(char), len, fp);
		scene->filename = data_root + scene->filename + ".bin"; 
		file_size += fread((void*)(scene->R), sizeof(float), 9, fp);
		file_size += fread((void*)(scene->K), sizeof(float), 9, fp);
		file_size += fread((void*)(&scene->height), sizeof(unsigned int), 1, fp);
		file_size += fread((void*)(&scene->width), sizeof(unsigned int), 1, fp); 
		file_size += fread((void*)(&len),    sizeof(unsigned int),   1, fp);
		scene->objects.resize(len);
		if (len > 0)
		{
			for (int i = 0; i <len; ++i)
			{
				Box3D box;
				file_size += fread((void*)(&(box.category)), sizeof(unsigned int),   1, fp);
				file_size += fread((void*)(box.base),        sizeof(float), 9, fp);
				file_size += fread((void*)(box.center),      sizeof(float), 3, fp);
				file_size += fread((void*)(box.coeff),       sizeof(float), 3, fp);
				box = processbox (box, context_pad, grid_size[1]);
				scene->objects[i] = box;

				int tarId = -1;
				if (this->phase_ == caffe::TRAIN)
				{
					uint8_t hasTarget = 0;
					file_size += fread((void*)(&hasTarget), sizeof(uint8_t),   1, fp);
					if (hasTarget > 0)
					{
						Box3Ddiff box_tar_diff;
						file_size += fread((void*)(box_tar_diff.diff), sizeof(float), 6, fp);
						tarId = target_boxes.size();
						target_boxes.push_back(box_tar_diff);
					}
				}
          	// push to back ground forground list by category 
          	// category 0 is negtive 
				if (box.category == 0 || this->phase_ == caffe::TEST) {
					scene->imgobj_neg.push_back(ImgObjInd(scenes.size(), i, tarId));
				}else{
					scene->imgobj_pos.push_back(ImgObjInd(scenes.size(), i, tarId));
				}

			}
		}
		shuffleData(scene, true, true);
		scenes.push_back(scene);
	}
	fclose(fp);
    std::cout<<"num_categories= "   << num_categories << std::endl;
    std::cout << "num of scenes = "   << scenes.size() << std::endl;
}

template <typename Dtype>
void Scene3DDataLayer<Dtype>::shuffleCategory(int category_id)
{
	caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
	shuffle(imgobj_pos_cates[category_id].begin(),imgobj_pos_cates[category_id].end(), prefetch_rng );
    cout<< "shuffle category "<< category_id << endl;
}

template <typename Dtype>
void Scene3DDataLayer<Dtype>::shuffleData(Scene3D* scene, bool neg, bool pos)
{
	caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
	if (pos){
      //std::cout<< "shuffle positive data" <<std::endl;
      shuffle (scene->imgobj_pos.begin(), scene->imgobj_pos.end(), prefetch_rng );
    }
    if (neg){
      //std::cout<< "shuffle neg data" <<std::endl;
      shuffle (scene->imgobj_neg.begin(), scene->imgobj_neg.end(), prefetch_rng );
    }
}

template <typename Dtype>
void Scene3DDataLayer<Dtype>::InternalThreadEntry() {
	
}

INSTANTIATE_CLASS(Scene3DDataLayer);
REGISTER_LAYER_CLASS(Scene3DData);
}
