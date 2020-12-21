/**/
#include "surface_field.h"
#include "marching_cubes.h"
#include "fUtility.h"
#include "vcg_inc.h"
#include <omp.h>

#include "wrap/ply/plylib.cpp"

/**************************************************************************
 *函数入口，主函数
 */
int main(int argc, char** argv)
{
	surface_field sur_sys;
	std::vector<vec3d> position;
	float radius_p, radius_h;
	vec3d space_min, space_max;

	radius_p = 0.1f; radius_h = 2*radius_p;
	space_min.set(-8);
	space_max.set(8, 12, 8);

	std::cout << "Input radius_p, radius_h, space_min(3f), space_max(3f)\n";
	std::cin >> radius_p >> radius_h >>
		space_min[0] >> space_min[1] >> space_min[2] >>
		space_max[0] >> space_max[1] >> space_max[2];

	sur_sys.surface_setup(space_min, space_max, radius_p, radius_h, radius_p/4);


	float iso=0.5f; //////////////

	std::vector<float> data;
	std::vector<float> result; std::vector<int> tris;
	GLTriMesh mesh;

	std::string maskstr;


	int MIN_FILE_NUM, MAX_FILE_NUM;
	std::cout << "Input min file number and max file number\n";
	std::cin >> MIN_FILE_NUM >> MAX_FILE_NUM;

	// log file
	char filestr[501]; sprintf_s(filestr, "Reconstruction log %d-%d ", MIN_FILE_NUM, MAX_FILE_NUM);
	string filename = filestr+get_date_time_string(true,false)+".log";
	std::ofstream f_clog(filename); f_clog.setf(std::ios::left);
	f_clog << "FileNumber;particleNum;surface_aniso();marchingCubes();vertexNum;triangleFaceNum;\n";
	double t_start;

	for( int file_i=MIN_FILE_NUM; file_i<MAX_FILE_NUM; ++file_i ) { // foreach file
		std::cout << file_i << std::endl;
		char name[50]; sprintf_s(name,"%d_6_c.pos",file_i);
		read_fluid_particles( name, data, &maskstr );
		int part_size = get_part_size( maskstr )/sizeof(float);

		int radiO1 = 50;
		real_t radiO = 0.1 / 1.5;
		real_t radiO2 = 7.5;

		position.resize(data.size() / part_size + radiO1 * radiO1*3);
		std::cout<<"partsize" << data.size() << endl;
		//position.resize(data.size()/part_size+ radiO1* radiO1);
		for(size_t i=0; i<position.size()- radiO1* radiO1*3; ++i){
			if(abs(data[i*part_size])>7.5 || abs(data[i*part_size + 1])>7.5 || abs(data[i*part_size + 2])>7.5)
				position[i].set(7.5,7.5,7.5);
			position[i].set(data[i*part_size],data[i*part_size+1],data[i*part_size+2]);
		}
		
		for (int k1 = 0; k1 < radiO1; k1++) {
			for (int k2 = 0; k2 < radiO1; k2++) {
				for (int k3 = 0; k3 < 1; k3++) {
					position[position.size()-(k1+1)*(k2+1)].set(radiO2- radiO*k1, radiO2 - radiO * k2, radiO2);
				}
			}
		}

		for (int k1 = 0; k1 < radiO1; k1++) {
			for (int k2 = 0; k2 < radiO1; k2++) {
				for (int k3 = 0; k3 < 1; k3++) {
					position[position.size() - (k1 + 1)*(k2 + 1)-250].set(radiO2 - radiO * k1, radiO2 - radiO * k2, radiO2- radiO);
				}
			}
		}

		for (int k1 = 0; k1 < radiO1; k1++) {
			for (int k2 = 0; k2 < radiO1; k2++) {
				for (int k3 = 0; k3 < 1; k3++) {
					position[position.size() - (k1 + 1)*(k2 + 1) - 500].set(radiO2 - radiO * k1, radiO2 - radiO * k2, radiO2 - radiO*3);
				}
			}
		}
		/*position[position.size() - 8].set(radiO2, radiO2, radiO2);
		position[position.size() - 7].set(radiO2, radiO2, radiO2 - radiO);
		position[position.size() - 6].set(radiO2, radiO2 - radiO, radiO2);
		position[position.size() - 5].set(radiO2 - radiO, radiO2, radiO2);
		position[position.size() - 4].set(radiO2 - radiO, radiO2, radiO2 - radiO);
		position[position.size() - 3].set(radiO2- radiO, radiO2- radiO, radiO2- radiO);
		position[position.size() - 2].set(radiO2- radiO, radiO2- radiO, radiO2);
		position[position.size() - 1].set(radiO2, radiO2- radiO, radiO2- radiO);*/

		f_clog.width(11); f_clog << file_i << ' ';
		f_clog.width(11); f_clog << data.size()/part_size << ' ';

		t_start = omp_get_wtime();
		sur_sys.surface_aniso( position );
		t_start = omp_get_wtime()-t_start;
		f_clog.width(11); f_clog << t_start << ' ';

		t_start = omp_get_wtime();
		int_t num = marchingCubes( result, iso,
			sur_sys.get_field(), sur_sys.get_grid_min().to<float>(), sur_sys.get_grid_num(), (float)sur_sys.get_cell_size() );
		t_start = omp_get_wtime()-t_start;
		f_clog.width(11); f_clog << t_start << ' ';

		tris.clear();
		for(size_t i=0; i<result.size()/3; ++i) tris.push_back(int(i));
		vcg_loadFromData(mesh, result, tris, true, false, false);
		sprintf_s(name,"%d_6c.ply",file_i);
		vcg_saveToPLY( mesh, name, true, true );

		f_clog.width(11); f_clog << mesh.VN() << ' ';
		f_clog.width(11); f_clog << mesh.FN() << ' ';

		f_clog << '\n';
	}
	return 0;
}

#if defined(_MSC_VER) && defined(_DEBUG)
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
class _LeakCheckStatic {
public:
	_LeakCheckStatic(){
		int tmpDbgFlag;
		tmpDbgFlag = _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG);
		tmpDbgFlag |= _CRTDBG_LEAK_CHECK_DF;
		_CrtSetDbgFlag(tmpDbgFlag);
		//_CrtSetBreakAlloc(418); // set break point at point of malloc
		char* p = new char[16]; // for test
		strcpy(p, "leak test @-@ ");
	}
} __leakObj;
#endif
