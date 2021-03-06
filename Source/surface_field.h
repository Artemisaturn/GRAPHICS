/* 
 * heliangliang, USTB, 2012.03.11, CopyRight Reserved
 * see "Reconstructing surfaces of particle-based fluids using anisotropic kernels (2010)"
*/

#ifndef _SURFACE_FIELD_
#define _SURFACE_FIELD_

#include"vec23.h"
//#include"wcsph.h"
#include"jama_svd.h"
#include"jama_eig.h"
#include<algorithm>
#include<vector>

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

using namespace TNT;
using namespace JAMA;

#define RE_TYPE float

#ifdef _WIN64
#	define INT_T long long
#else
#	define INT_T int
#endif

class surface_field {
public:
	inline surface_field() { }
	inline ~surface_field() { }
	void surface_setup( vec3d min, vec3d max, double r_p, double r_h, double cell_size );
	const std::vector<RE_TYPE>& surface_isotro( const std::vector<vec3d>& position );
	const std::vector<RE_TYPE>& surface_aniso( const std::vector<vec3d>& position );

	inline const vec3d& get_grid_min() const { return field_grid_min; }
	inline const vec3<INT_T>& get_grid_num() const { return field_grid_cell_num; }
	inline double get_cell_size() const { return field_grid_cell_size; }
	inline const std::vector<RE_TYPE>& get_field() const { return field_grid; }

private:
	inline double ker_spline(double r);
	inline double weight_wij(double r); // Equation (11)
	void grid_insert(const std::vector<vec3d>& position);
	void grid_neighbors(const vec3d& p, INT_T* r);

	// field grid
	std::vector<RE_TYPE>	field_grid;
	INT_T			field_grid_num;
	vec3d			field_grid_min;
	double			field_grid_cell_size;
	vec3<INT_T>		field_grid_cell_num;
	vec3d			field_grid_divisor;

	// parameters
	double	radius_h; // the h in the SPH equation
	double	volume_p; // particle's volume
	double  h_field;// the radius in Equation (11)
	double	k_r;	// Equation (15)
	double	k_s;	// Equation (15)
	double  k_n;	// Equation (15)
	INT_T	N_e;	// N_Epsilon, Equation (15)
	double  lambda; // Lambda, Equation (6)

	// particle grid
	INT_T			particle_num;
	std::vector<INT_T>	p_next;
	std::vector<INT_T>	grid_first;
	vec3<INT_T>		grid_cell_num;
	vec3d		grid_divisor;
	vec3d		grid_min;

};

#endif // #ifndef _SURFACE_FIELD_
