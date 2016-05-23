#include <fstream>
#include <iostream>
#include "caffe/util/dss.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "caffe/caffe.hpp"

using namespace std;
using namespace caffe;

int main()
{
	int count1 = 0, count2 = 0;
	string file_list = "/home/ganyk/nips2016/my_train.list";
	std::cout<<"loading file "<<file_list<<"\n";
	FILE* fp = fopen(file_list.c_str(),"rb");
	FILE* fp1 = fopen("file.txt", "w");
	if (fp == NULL) { std::cout<<"fail to open file: "<< file_list << std::endl; exit(1); }
	size_t file_size;
	int num = 0;
	while (feof(fp) == 0) {
		count1 = 0;
		count2 = 0;
		Scene3D<float>* scene = new Scene3D<float>();
		unsigned int len = 0;
		file_size += fread((void*)(&len), sizeof(unsigned int), 1, fp);    
		if (len==0) break;
		scene->filename.resize(len);
		if (len>0) 
			file_size += fread((void*)(scene->filename.data()), sizeof(char), len, fp);
		scene->filename = scene->filename + ".bin"; 
		//fprintf(fp1, "%s\n", scene->filename.c_str());
        //std::cout<<scene->filename<<std::endl;
		file_size += fread((void*)(scene->R), sizeof(float), 9, fp);
		file_size += fread((void*)(scene->K), sizeof(float), 9, fp);
		file_size += fread((void*)(&scene->height), sizeof(unsigned int), 1, fp);
		file_size += fread((void*)(&scene->width), sizeof(unsigned int), 1, fp); 
		file_size += fread((void*)(&len),    sizeof(unsigned int),   1, fp);
		scene->objects.resize(len);
		
		if (len > 0) {
			for (int i = 0; i < len; ++i){
				struct Box3D box;
				file_size += fread((void*)(&(box.category)), sizeof(unsigned int),   1, fp);
				file_size += fread((void*)(box.base),        sizeof(float), 9, fp);
				file_size += fread((void*)(box.center),      sizeof(float), 3, fp);
				file_size += fread((void*)(box.coeff),       sizeof(float), 3, fp);
				file_size += fread((void*)(box.box2d),		 sizeof(unsigned int),   4, fp);
				/*fprintf(fp1, "base: [%f %f %f; %f %f %f; %f %f %f] center: [%f %f %f] coeff: [%f %f %f]\n", 
					box.base[0], box.base[1], box.base[2], box.base[3], box.base[4], box.base[5], box.base[6], box.base[7], box.base[8],
					box.center[0], box.center[1], box.center[2], box.coeff[0], box.coeff[1], box.coeff[2]);*/				
				box = processbox (box, 3, 30);
				scene->objects[i] = box;

          //num_categories = max(num_categories, box.category);
          // read target box if exist
				if (box.category != 0)
					count1++;
				else
					count2++;
				uint8_t hasTarget = 0;
				file_size += fread((void*)(&hasTarget), sizeof(uint8_t),   1, fp);
				if (hasTarget > 0) {
					struct Box3Ddiff box_tar_diff;
					file_size += fread((void*)(box_tar_diff.diff), sizeof(float), 6, fp);
				}

			}
		}
		if (count1 == 0)
			fprintf(fp1, "%d\n", num);
		num++;

		scene->free();
	}

	//cout << num << endl;
	//cout << count1 << "   " << count2 << endl;
	fclose(fp);
	fclose(fp1);
	return 0;
}