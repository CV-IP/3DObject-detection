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
	context_pad = scene3d_param.context_pad();
	encode_type = scene3d_param.encode_type();
	scale = scene3d_param.scale();
	num_categories = scene3d_param.num_categories();

	imgobj_pos_cates.resize(num_categories);

	//get the batch size
	batch_size.push_back(scene3d_param.batch_size(0));
	batch_size.push_back(scene3d_param.batch_size(1));

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

	scenes_id_ = 0;

	std::cout << "loading file " << file_list << "\n";
	FILE* fp = fopen(file_list.c_str(), "rb");
	if (fp == NULL) { std::cout << "fail to open file: " << file_list << std::endl; exit(1); }
	size_t file_size;
	while (feof(fp) == 0)
	{
		Scene3D* scene = new Scene3D();
		scene->counter_pos = 0;
		scene->counter_neg = 0;

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
				file_size += fread((void*)(box.box2d),		 sizeof(unsigned int),   4, fp);
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
    //std::cout<<"num_categories= "   << num_categories << std::endl;
    //std::cout << "num of scenes = "   << scenes.size() << std::endl;

    //reshape the blobs
    vector<int> data_size;
    data_size.push_back(sum(batch_size));
    for (int i = 0; i < grid_size.size(); i++)
    	data_size.push_back(grid_size[i]);
    this->prefetch_data_.Reshape(data_size);
    top[0]->Reshape(data_size);

    vector<int> label_size(1, sum(batch_size));
    this->prefetch_label_.Reshape(label_size);
    top[1]->Reshape(label_size);

    vector<int> bb2d_size;
    bb2d_size.push_back(sum(batch_size));
    bb2d_size.push_back(5);
    this->prefetch_bb2d_proj_.Reshape(bb2d_size);
    top[2]->Reshape(bb2d_size);

    vector<int> attention_bb_size;
    attention_bb_size.push_back(sum(batch_size));
    attention_bb_size.push_back(5);
    this->prefetch_attention_bb_.Reshape(attention_bb_size);
    top[3]->Reshape(attention_bb_size);

    vector<int> bb3d_diff_size;
    bb3d_diff_size.push_back(sum(batch_size));
    bb3d_diff_size.push_back(6);
    this->prefetch_bb3d_diff_.Reshape(bb3d_diff_size);
    top[4]->Reshape(bb3d_diff_size);

    vector<int> bb3d_param_size(1, bb_param_weight.size());
    this->prefetch_bb3d_param_.Reshape(bb3d_param_size);
    top[5]->Reshape(bb3d_param_size);

    Dtype* bb3d_param_data = this->prefetch_bb3d_param_.mutable_cpu_data();
    for (int i = 0; i < this->prefetch_bb3d_param_.count(); i++)
    {
    	bb3d_param_data[i] = bb_param_weight[i];
    }
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

//get the distance between two bounding boxes
int get_dist(int scene_id, int obj_id1, int obj_id2)
{
	int x1 = scenes[scene_id].objects[obj_id1].box2d[0];
	int y1 = scenes[scene_id].objects[obj_id1].box2d[1];
	int x2 = scenes[scene_id].objects[obj_id2].box2d[0];
	int y2 = scenes[scene_id].objects[obj_id2].box2d[1];

	return (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1);
}

float getIOU(int scene_id, int obj_id1, int obj_id2)
{
	int obj1_x1 = scenes[scene_id].objects[obj_id1].box2d[0];
	int obj1_y1 = scenes[scene_id].objects[obj_id1].box2d[1];
	int obj1_x2 = scenes[scene_id].objects[obj_id1].box2d[0] + scenes[scene_id].objects[obj_id1].box2d[2];
	int obj1_y2 = scenes[scene_id].objects[obj_id1].box2d[1] + scenes[scene_id].objects[obj_id1].box2d[3];
	int obj2_x1 = scenes[scene_id].objects[obj_id2].box2d[0];
	int obj2_y1 = scenes[scene_id].objects[obj_id2].box2d[1];
	int obj2_x2 = scenes[scene_id].objects[obj_id2].box2d[0] + scenes[scene_id].objects[obj_id2].box2d[2];
	int obj2_y2 = scenes[scene_id].objects[obj_id2].box2d[1] + scenes[scene_id].objects[obj_id2].box2d[3];

	int area1 = (obj1_y2 - obj1_y1) * (obj1_x2 - obj1_x1);
	int area2 = (obj2_y2 - obj2_y1) * (obj2_x2 - obj2_x1);
	int intersection_area;

	if (obj1_x1 <= obj2_x1 && obj2_x1 <= obj1_x2 &&
		obj1_y1 <= obj2_y1 && obj2_y1 <= obj1_y2)
	{
		if (obj1_y2 <= obj2_y2 && obj1_x2 <= obj2_x2)
		{
			intersection_area = (obj1_y2 - obj2_y1) * (obj1_x2 - obj2_x1);
		}
		else if (obj1_y2 > obj2_y2 && obj1_x2 <= obj2_x2)
		{
			intersection_area = (obj2_y2 - obj2_y1) * (obj1_x2 - obj2_x1);
		}
		else if (obj1_y2 <= obj2_y2 && obj1_x2 > obj2_x2)
		{
			intersection_area = (obj1_y2 - obj2_y1) * (obj2_x2 - obj2_x1);
		}
		else
		{
			intersection_area = (obj2_y2 - obj2_y1) * (obj2_x2 - obj2_x1);
		}
	}
	else if (obj2_x1 <= obj1_x1 && obj1_x1 <= obj2_x2 &&
		obj2_y1 <= obj1_y1 && obj1_y1 <= obj2_y2)
	{
		if (obj2_y2 <= obj1_y2 && obj2_x2 <= obj1_x2)
		{
			intersection_area = (obj2_y2 - obj1_y1) * (obj2_x2 - obj1_x1);
		}	
		else if (obj2_y2 > obj1_y2 && obj2_x2 <= obj1_x2)
		{
			intersection_area = (obj1_y2 - obj1_y1) * (obj2_x2 - obj1_x1);
		}
		else if (obj2_y2 <= obj1_y2 && obj2_x2 > obj1_x2)
		{
			intersection_area = (obj2_y2 - obj1_y1) * (obj1_x2 - obj1_x1);
		}	
		else
		{
			intersection_area = (obj1_y2 - obj1_y1) * (obj1_x2 - obj1_x1);
		}
	}

	return intersection_area / float(area1 + area2 - intersection_area);
}

template <typename Dtype>
void Scene3DDataLayer<Dtype>::compute_attention_area(int scene_id, int obj_id, Dtype* attention_bb_data)
{
	vector<int> dists;
	vector<int> idx;

	idx.push_back(obj_id);
	dists.push_back(0);
	for (int i = 0; i < scenes[scene_id].objects.size(); i++)
	{
		if (i == obj_id)
			continue;
		int j;
		for (j = 0; j < idx.size(); j++)
		{
			if (getIOU(scene_id, idx[j], i) > 0.4)
				break;
		}

		if (j == idx.size())
		{
			int count = 0;
			int dist = get_dist(scene_id, obj_id, i);
			while (dist[count] < dist) count++;
			idx.insert(idx.begin() + count, i);
			dists.insert(dists.begin() + count, i);
		}
	}

	int min_x = 1000;
	int min_y = 1000;
	int max_x = 0;
	int max_y = 0;

	for (int i = 0; i < idx.size() && i < 50; i++)
	{
		int x1 = scenes[scene_id].objects[idx[i]].box2d[0];
		int y1 = scenes[scene_id].objects[idx[i]].box2d[1];
		int x2 = scenes[scene_id].objects[idx[i]].box2d[0] + scenes[scene_id].objects[idx[i]].box2d[2];
		int y2 = scenes[scene_id].objects[idx[i]].box2d[1] + scenes[scene_id].objects[idx[i]].box2d[3];

		if (x1 < min_x) min_x = x1;
		if (y1 < min_y) min_y = y1;
		if (x2 > max_x) max_x = x2;
		if (y2 > max_y) max_y = y2;
	}

	attention_bb_data[0] = Dtype(min_x);
	attention_bb_data[1] = Dtype(min_y);
	attention_bb_data[2] = Dtype(max_x - min_x);
	attention_bb_data[3] = Dtype(max_y - min_y);
}

template <typename Dtype>
void Scene3DDataLayer<Dtype>::InternalThreadEntry() {

	CPUTimer batch_timer;
	batch_timer.Start();
	vector<Scene3D*> chosen_scenes;
    vector<int> chosen_box_id;
    int totalpos, totalneg, batch_count = 0;

    if (batch_size.size() == 1 || this->phase_ == caffe::TEST)
    {
    	totalpos = 0;
    	totalneg = sum(batch_size);
    }
    else
    {
    	totalneg = batch_size[0];
    	totalpos = batch_size[1];
    }

    Dtype* data_gpu = this->prefetch_data_.mutable_gpu_data();
    Dtype* label_data = this->prefetch_data_.mutable_cpu_data();
    Dtype* bb2d_data = this->prefetch_bb2d_proj_.mutable_cpu_data();
    Dtype* attention_bb_data = this->prefetch_attention_bb_.mutable_cpu_data();
    Dtype* bb3d_diff_data = this->prefetch_bb3d_diff_.mutable_cpu_data();

    //get the negative boxes from two images
    for (int i = scenes_id_; i < scenes_id_ + 2; i = (i + 1) % scenes.size())
    {
    	for (int j = 0; j < totalneg / 2; j++)
    	{
      		int objId = scenes[i]->imgobj_neg[scenes[i]->counter_neg].ObjId;
      		chosen_scenes.push_back(scenes[i]);
      		chosen_box_id.push_back(objId);
      		Box3D box = scenes[i]->objects[objId];

      		label_data[batch_count] = Dtype(box.category);

      		bb2d_data += this->prefetch_bb2d_proj_.offset(batch_count);
      		bb2d_data[0] = Dtype(i - scenes_id_);
      		bb2d_data[1] = Dtype(box.box2d[0]);
      		bb2d_data[2] = Dtype(box.box2d[1]);
      		bb2d_data[3] = Dtype(box.box2d[2]);
      		bb2d_data[4] = Dtype(box.box2d[3]);

      		attention_bb_data += this->prefetch_attention_bb_.offset(batch_count);
      		attention_bb_data[0] = Dtype(i - scenes_id_);
      		compute_attention_area(i, objId, attention_bb_data + 1);

      		scenes[i]->counter_neg++;
      		++batch_count;

      		if (scenes[i]->counter_neg >= scenes[i]->imgobj_neg.size())
      		{
      			scenes[i]->counter_neg = 0;
      			if (this->phase_ == caffe::TRAIN)
      			{
      				shuffleData(scenes[i], true, false);
      			}
      		}
    	}
    }

    //get the postive boxes from two images
    for (int i = scenes_id_; i < scenes_id_ + 2; i = (i + 1) % scenes.size())
    {
    	for (int j = 0; j < totalpos / 2; j++)
    	{
      		int objId = scenes[i]->imgobj_pos[scenes[i]->counter_pos].ObjId;
      		int tarId = scenes[i]->imgobj_pos[scenes[i]->counter_pos].TarId
      		chosen_scenes.push_back(scenes[i]);
      		chosen_box_id.push_back(objId);
      		Box3D box = scenes[i]->objects[objId];

      		label_data[batch_count] = Dtype(box.category);

      		bb2d_data += this->prefetch_bb2d_proj_.offset(batch_count);
      		bb2d_data[0] = Dtype(i - scenes_id_);
      		bb2d_data[1] = Dtype(box.box2d[0]);
      		bb2d_data[2] = Dtype(box.box2d[1]);
      		bb2d_data[3] = Dtype(box.box2d[2]);
      		bb2d_data[4] = Dtype(box.box2d[3]);

      		attention_bb_data += this->prefetch_attention_bb_.offset(batch_count);
      		attention_bb_data[0] = Dtype(i - scenes_id_);
      		compute_attention_area(i, objId, attention_bb_data + 1);

      		bb3d_diff_data += this->prefetch_bb3d_diff_.offset(batch_count);
      		for (int cid = 0; cid < 6; cid++)
      		{
      			bb3d_diff_data[cid] = target_boxes[tarId].diff[cid];
      		}

      		scenes[i]->counter_pos++;
      		++batch_count;

      		if (scenes[i]->counter_pos >= scenes[i]->imgobj_pos.size())
      		{
      			scenes[i]->counter_pos = 0;
      			if (this->phase_ == caffe::TRAIN)
      			{
      				shuffleData(scenes[i], false, true);
      			}
      		}
    	}
    }
    // now, the list of scenes and wanted boxes are in chosen_scenes, compute the TSDF and store the data in dataCPU
    compute_TSDF(&chosen_scenes, &chosen_box_id, data_gpu, grid_size, encode_type, scale); 

    scenes_id_ = (scenes_id_ + 2) % scenes.size();
    batch_timer.Stop();
    DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
}

INSTANTIATE_CLASS(Scene3DDataLayer);
REGISTER_LAYER_CLASS(Scene3DData);
}
