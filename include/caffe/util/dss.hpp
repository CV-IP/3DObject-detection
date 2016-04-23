#ifndef CAFFE_UTIL_DSS_HPP_
#define CAFFE_UTIL_DSS_HPP_

#include "caffe/common.hpp"

namespace caffe {

  struct ImgObjInd{
    int ImageId;
    int ObjId; 
    int TarId;
    ImgObjInd(int i,int j,int k){
      ImageId = i;
      ObjId = j;
      TarId = k;
    }
  };

  struct RGBDpixel{
    uint8_t R;
    uint8_t G;
    uint8_t B;
    uint8_t D;
    uint8_t D_;
  };

  struct Box3D{
    unsigned int category;
    float base[9];
    float center[3];
    float coeff[3];
  };

  struct Box2D{
    unsigned int category;
    float tblr[4];
  };

  struct Box3Ddiff{
    float diff[6];
    float diff2d[4];
    int oreintation;
  };

  struct mesh_meta{
  // x : left right 
  // y : height 
  // z : depth
    std::string mesh_file;
  float base[9];// oreintation 
  float center[3];
  float coeff[3];
};

Box3D processbox (Box3D box, float context_pad, int tsdf_size);

enum Scene3DType { RGBD, Render, Mesh };

class Scene3D{
public:
  // defined in .list file
  std::vector<mesh_meta> mesh_List;

  std::string filename;
  std::string seqname;

  float K[9];
  float R[9];
  unsigned int width;
  unsigned int height;
  unsigned int len_pcIndex;
  unsigned int len_beIndex;
  unsigned int len_beLinIdx;
  std::vector<Box3D> objects;
  std::vector<Box2D> objects_2d_tight;
  std::vector<Box2D> objects_2d_full;

  std::vector<ImgObjInd> imgobj_pos;
  std::vector<ImgObjInd> imgobj_neg;

  bool GPUdata;
  Scene3DType DataType;
  // defined in .data file
  unsigned int* grid_range;
  float* begin_range;
  float grid_delta;
  RGBDpixel* RGBDimage;
  unsigned int* beIndex;
  unsigned int* beLinIdx;
  unsigned int* pcIndex;
  float* XYZimage;
  float* K_GPU;
  float* R_GPU;

  

  //Scene3D(): RGBDimage(NULL), beIndex(NULL), pcIndex(NULL), beLinIdx(NULL),XYZimage(NULL), grid_range(NULL), begin_range(NULL),K_GPU(NULL),R_GPU(NULL),GPUdata(false),isMesh(false){};
  Scene3D(){
    RGBDimage = NULL;
    beIndex = NULL;
    pcIndex = NULL;
    beLinIdx = NULL;
    XYZimage = NULL;
    grid_range = NULL;
    begin_range = NULL;
    K_GPU = NULL;
    R_GPU = NULL;

    GPUdata = false;
    DataType = RGBD;
  }

  void compute_xyz() {
    XYZimage = new float[width*height*3];
    //printf("scene->K:%f,%f,%f\n%f,%f,%f\n%f,%f,%f\n",K[0],K[1],K[2],K[3],K[4],K[5],K[6],K[7],K[8]);
    for (int ix = 0; ix < width; ix++){
      for (int iy = 0; iy < height; iy++){
        float depth = float(*((uint16_t*)(&(RGBDimage[iy + ix * height].D))))/1000.0;
        
          // project the depth point to 3d
        float tdx = (float(ix + 1) - K[2]) * depth / K[0];
        float tdz =  - (float(iy + 1) - K[5]) * depth / K[4];
        float tdy = depth;

        XYZimage[3 * (iy + ix * height) + 0] = R[0] * tdx + R[1] * tdy + R[2] * tdz;
        XYZimage[3 * (iy + ix * height) + 1] = R[3] * tdx + R[4] * tdy + R[5] * tdz;
        XYZimage[3 * (iy + ix * height) + 2] = R[6] * tdx + R[7] * tdy + R[8] * tdz;
      }

    }
  }

  void compute_xyzGPU();

  void loadData2XYZimage(){
  		//this function only support RGBD data type
    this ->load();
    this -> cpu2gpu();
    this -> compute_xyzGPU();
  }

  int load(){
    int filesize =0;
    if (RGBDimage==NULL||beIndex==NULL||pcIndex==NULL||XYZimage==NULL){
      //std::cout<< "loading image "<< filename<<std::endl;
      free();
      FILE* fp = fopen(filename.c_str(),"rb");
      if (fp==NULL) { std::cout<<"in load() :fail to open file: "<<filename<<std::endl; exit(EXIT_FAILURE); }
      grid_range = new unsigned int[3];
      filesize += fread((void*)(grid_range), sizeof(unsigned int), 3, fp);
      
      begin_range = new float[3];
      filesize += fread((void*)(begin_range), sizeof(float), 3, fp);
      filesize += fread((void*)(&grid_delta), sizeof(float), 1, fp);

      RGBDimage = new RGBDpixel[width*height];
      filesize += fread((void*)(RGBDimage), sizeof(RGBDpixel), width*height, fp);

      filesize +=  fread((void*)(&len_beIndex), sizeof(unsigned int), 1, fp);
      beIndex   = new unsigned int [len_beIndex];
      filesize += fread((void*)(beIndex), sizeof(unsigned int), len_beIndex, fp);

      filesize +=  fread((void*)(&len_beLinIdx), sizeof(unsigned int), 1, fp);
      beLinIdx  = new unsigned int [len_beLinIdx];
      filesize += fread((void*)(beLinIdx), sizeof(unsigned int), len_beLinIdx, fp);

      filesize += fread((void*)(&len_pcIndex), sizeof(unsigned int), 1, fp);
      pcIndex   = new unsigned int [len_pcIndex];
      filesize += fread((void*)(pcIndex), sizeof(unsigned int), len_pcIndex, fp);
      fclose(fp);

      GPUdata = false;
    }
    return filesize;
  }

  void cpu2gpu();

  void free();

  ~Scene3D(){
    free();
  }
};

void compute_TSDF (std::vector<Scene3D*> *chosen_scenes_ptr, std::vector<int> *chosen_box_id, 
                    float* datamem, std::vector<int> grid_size, int encode_type, float scale);
void compute_TSDF_Space(Scene3D* scene , Box3D SpaceBox, float* tsdf_data_GPU, 
                    std::vector<int> grid_size, int encode_type, float scale);

}

#endif