#include <cmath>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include "caffe/util/dss.hpp"

namespace caffe {

int sum(std::vector<int> dim)
{
  return accumulate(dim.begin(), dim.end(), 0);
}

Box3D processbox (Box3D box, float context_pad, int tsdf_size)
{
  if (context_pad > 0){
    float context_scale = float(tsdf_size) / (float(tsdf_size) - 2*context_pad);
    box.coeff[0] = box.coeff[0] * context_scale;
    box.coeff[1] = box.coeff[1] * context_scale;
    box.coeff[2] = box.coeff[2] * context_scale;
  }
       // change the oreintation 
  if (box.base[1] < 0){
    box.base[0] = -1*box.base[0];
    box.base[1] = -1*box.base[1];
    box.base[2] = -1*box.base[2];
  }
  if (box.base[4] < 0){
    box.base[3] = -1*box.base[3];
    box.base[4] = -1*box.base[4];
    box.base[5] = -1*box.base[5];
  }

  if(box.base[1]<box.base[4]){
          // swap first two row 
    float tmpbase[3];
    tmpbase[0] = box.base[0];
    tmpbase[1] = box.base[1];
    tmpbase[2] = box.base[2];

    box.base[0] = box.base[3];
    box.base[1] = box.base[4];
    box.base[2] = box.base[5];

    box.base[3] = tmpbase[0];
    box.base[4] = tmpbase[1];
    box.base[5] = tmpbase[2];
    float tmpcoeff =  box.coeff[0];
    box.coeff[0] = box.coeff[1];
    box.coeff[1] = tmpcoeff;
  }
  return box;
}

}