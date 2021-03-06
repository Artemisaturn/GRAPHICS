/* 
 * heliangliang, USTB, 2012.03.05, CopyRight Reserved
*/

#ifndef _MARCHING_CUBES_H_
#define _MARCHING_CUBES_H_

#include "vec23.h"
#include "SphBase.h"

#include<vector>
#include<iostream>
#include<string>
#include<fstream>

int_t marchingCubes( std::vector<float>& result, float iso,
	const std::vector<float>& cubeGrid, vec3f min, veci_t cubeGridRes, float cell_size);


#endif // #ifndef _MARCHING_CUBES_H_
