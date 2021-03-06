// fUtility.h
#ifndef _FUTILITY_H_
#define _FUTILITY_H_

#include "SphBase.h"
#include "gl_inc.h"
#include "il_inc.h"
#include <fstream>
#include <cassert>
#include "json/json.h"


// use these macros to select the components of particles to be writed
#define WRITE_PARTICLE_position(file, p)	 file.write((const char*)&p.position[0], sizeof(vec_t))
//#define WRITE_PARTICLE_velocity(file, p)	 file.write((const char*)&p.velocity[0], sizeof(vec_t))
//#define WRITE_PARTICLE_color(file, p)		 file.write((const char*)&p.color[0], 4*sizeof(float))
//#define WRITE_PARTICLE_density(file, p)		 file.write((const char*)&p.density, sizeof(real_t))
//#define WRITE_PARTICLE_presure(file, p)		 file.write((const char*)&p.presure, sizeof(real_t))
//#define WRITE_PARTICLE_acceleration(file, p) file.write((const char*)&p.acceleration[0], sizeof(vec_t))
#ifdef WC_TIMEADAPTIVE
//#	define WRITE_PARTICLE_dt(file, p)		 file.write((const char*)&p.dt, sizeof(real_t));
#endif

// num byte of one particle wrtied
inline int get_part_size_static(){
	static int part_size=-1;
	if( part_size<=0 ) {
		part_size = 0;
	#ifdef WRITE_PARTICLE_position
		part_size += sizeof(vec_t);
	#endif
	#ifdef WRITE_PARTICLE_velocity
		part_size += sizeof(vec_t);
	#endif
	#ifdef WRITE_PARTICLE_color
		part_size += 4*sizeof(float);
	#endif
	#ifdef WRITE_PARTICLE_density
		part_size += sizeof(real_t);
	#endif
	#ifdef WRITE_PARTICLE_presure
		part_size += sizeof(real_t);
	#endif
	#ifdef WRITE_PARTICLE_acceleration
		part_size += sizeof(vec_t);
	#endif
	#ifdef WC_TIMEADAPTIVE
		#ifdef WRITE_PARTICLE_dt
		part_size += sizeof(real_t);
		#endif
	#endif
	}
	return part_size;
}

inline int get_part_size(const std::string& mask){
	int part_size = 0;
	if( mask.find("position")!=std::string::npos ) part_size += sizeof(vec_t);
	if( mask.find("velocity")!=std::string::npos ) part_size += sizeof(vec_t);
	if( mask.find("color")!=std::string::npos ) part_size += 4*sizeof(float);
	if( mask.find("density")!=std::string::npos ) part_size += sizeof(real_t);
	if( mask.find("presure")!=std::string::npos ) part_size += sizeof(real_t);
	if( mask.find("acceleration")!=std::string::npos ) part_size += sizeof(vec_t);
	if( mask.find("dt")!=std::string::npos ) part_size += sizeof(real_t);
	return part_size;
}

// a string tell that which component is writed
inline std::string get_part_write_mask(){
	static std::string write_mask="-";
	if( write_mask=="-" ) {
		write_mask = "<";
	#ifdef WRITE_PARTICLE_position
		write_mask += " position ";
	#endif
	#ifdef WRITE_PARTICLE_velocity
		write_mask += " velocity ";
	#endif
	#ifdef WRITE_PARTICLE_color
		write_mask += " color ";
	#endif
	#ifdef WRITE_PARTICLE_density
		write_mask += " density ";
	#endif
	#ifdef WRITE_PARTICLE_presure
		write_mask += " presure ";
	#endif
	#ifdef WRITE_PARTICLE_acceleration
		write_mask += " acceleration ";
	#endif
	#ifdef WC_TIMEADAPTIVE
		#ifdef WRITE_PARTICLE_dt
		write_mask += " dt ";
		#endif
	#endif
		write_mask += ">";
	}

	return write_mask;
}

/*
position file format:
number of particle N(int)
N lines position data, x,y,z
*/

/*
inline void write_fluid_particles(const char* filename1, const char* filename2, const SphBase& sphObj, int idx=0)
{
	assert( idx>=0 && idx<sphObj.getNumFluids() );

	const std::vector<SphBase::FluidPart>& ps = sphObj.getFluidParticles(idx);
	int num = ps.size();
	real_t beta;
	std::vector<int> beta1_i;
	std::vector<int> beta2_i;

	for (int i = 0; i < num; ++i)
	{
		beta = ps[i].beta;
		if (beta == 1.0) {
			beta1_i.push_back(i);
		}
		else {
			beta2_i.push_back(i);
		}
	}
	std::ofstream outfile;
	outfile.open(filename1, std::fstream::binary);
	if (!outfile) { // open failed
		outfile.close();
		std::cout << "write_pos(): open file failed!\n";
		return;
	}

	// line 1, log file
	outfile << "for parameters and scene info, please refer to \""
		<< sphObj.getLogFileName() << "\"" << std::endl;
	// line 2, components writed
	outfile << "components writed: " << get_part_write_mask() << std::endl;
	// line 3, frame number & fluid index
	outfile << sphObj.getFrameNumber() << " (frame number), fluid index: " << idx << std::endl;
	// line 4, size
	outfile << beta1_i.size() << " (num particles)" << std::endl;

	int i = 0;
	for (int index = 0; index < beta1_i.size(); ++index) {
		i = beta1_i[index];
		#ifdef WRITE_PARTICLE_position
				WRITE_PARTICLE_position(outfile, ps[i]);
		#endif
		#ifdef WRITE_PARTICLE_velocity
				WRITE_PARTICLE_velocity(outfile, ps[i]);
		#endif
		#ifdef WRITE_PARTICLE_color
				WRITE_PARTICLE_color(outfile, ps[i]);
		#endif
		#ifdef WRITE_PARTICLE_density
				WRITE_PARTICLE_density(outfile, ps[i]);
		#endif
		#ifdef WRITE_PARTICLE_presure
				WRITE_PARTICLE_presure(outfile, ps[i]);
		#endif
		#ifdef WRITE_PARTICLE_acceleration
				WRITE_PARTICLE_acceleration(outfile, ps[i]);
		#endif
		#ifdef WC_TIMEADAPTIVE
			#ifdef WRITE_PARTICLE_dt
					WRITE_PARTICLE_dt(outfile, ps[i]);
			#endif
		#endif
	}

	outfile.close();

	//beta2
	outfile.open(filename2, std::fstream::binary);
	if (!outfile) { // open failed
		outfile.close();
		std::cout << "write_pos(): open file failed!\n";
		return;
	}

	// line 1, log file
	outfile << "for parameters and scene info, please refer to \""
		<< sphObj.getLogFileName() << "\"" << std::endl;
	// line 2, components writed
	outfile << "components writed: " << get_part_write_mask() << std::endl;
	// line 3, frame number & fluid index
	outfile << sphObj.getFrameNumber() << " (frame number), fluid index: " << idx << std::endl;

	// line 4, size
	outfile << beta2_i.size() << " (num particles)" << std::endl;

	i = 0;
	for (int index = 0; index < beta2_i.size(); ++index) {
		i = beta2_i[index];
		#ifdef WRITE_PARTICLE_position
				WRITE_PARTICLE_position(outfile, ps[i]);
		#endif
		#ifdef WRITE_PARTICLE_velocity
				WRITE_PARTICLE_velocity(outfile, ps[i]);
		#endif
		#ifdef WRITE_PARTICLE_color
				WRITE_PARTICLE_color(outfile, ps[i]);
		#endif
		#ifdef WRITE_PARTICLE_density
				WRITE_PARTICLE_density(outfile, ps[i]);
		#endif
		#ifdef WRITE_PARTICLE_presure
				WRITE_PARTICLE_presure(outfile, ps[i]);
		#endif
		#ifdef WRITE_PARTICLE_acceleration
				WRITE_PARTICLE_acceleration(outfile, ps[i]);
		#endif
		#ifdef WC_TIMEADAPTIVE
			#ifdef WRITE_PARTICLE_dt
					WRITE_PARTICLE_dt(outfile, ps[i]);
			#endif
		#endif
	}

	outfile.close();

	/*
	for(size_t i=0; i<ps.size(); ++i){
	#ifdef WRITE_PARTICLE_position
		WRITE_PARTICLE_position(outfile, ps[i]);
	#endif
	#ifdef WRITE_PARTICLE_velocity
		WRITE_PARTICLE_velocity(outfile, ps[i]);
	#endif
	#ifdef WRITE_PARTICLE_color
		WRITE_PARTICLE_color(outfile, ps[i]);
	#endif
	#ifdef WRITE_PARTICLE_density
		WRITE_PARTICLE_density(outfile, ps[i]);
	#endif
	#ifdef WRITE_PARTICLE_presure
		WRITE_PARTICLE_presure(outfile, ps[i]);
	#endif
	#ifdef WRITE_PARTICLE_acceleration
		WRITE_PARTICLE_acceleration(outfile, ps[i]);
	#endif
	#ifdef WC_TIMEADAPTIVE
		#ifdef WRITE_PARTICLE_dt
		WRITE_PARTICLE_dt(outfile, ps[i]);
		#endif
	#endif
	}	
}
*/
inline void write_fluid_particles(const char* filename, const SphBase& sphObj, int idx = 0)
{
	assert(idx >= 0 && idx < sphObj.getNumFluids());

	std::ofstream outfile;
	outfile.open(filename, std::fstream::binary);
	if (!outfile) { // open failed
		outfile.close();
		std::cout << "write_pos(): open file failed!\n";
		return;
	}

	// line 1, log file
	outfile << "for parameters and scene info, please refer to \""
		<< sphObj.getLogFileName() << "\"" << std::endl;
	// line 2, components writed
	outfile << "components writed: " << get_part_write_mask() << std::endl;
	// line 3, frame number & fluid index
	outfile << sphObj.getFrameNumber() << " (frame number), fluid index: " << idx << std::endl;
	// line 4, size
	const std::vector<SphBase::FluidPart>& ps = sphObj.getFluidParticles(idx);
	outfile << ps.size() << " (num particles)" << std::endl;

	for (size_t i = 0; i < ps.size(); ++i) {
		#ifdef WRITE_PARTICLE_position
				WRITE_PARTICLE_position(outfile, ps[i]);
		#endif
		#ifdef WRITE_PARTICLE_velocity
				WRITE_PARTICLE_velocity(outfile, ps[i]);
		#endif
		#ifdef WRITE_PARTICLE_color
				WRITE_PARTICLE_color(outfile, ps[i]);
		#endif
		#ifdef WRITE_PARTICLE_density
				WRITE_PARTICLE_density(outfile, ps[i]);
		#endif
		#ifdef WRITE_PARTICLE_presure
				WRITE_PARTICLE_presure(outfile, ps[i]);
		#endif
		#ifdef WRITE_PARTICLE_acceleration
				WRITE_PARTICLE_acceleration(outfile, ps[i]);
		#endif
		#ifdef WC_TIMEADAPTIVE
		#ifdef WRITE_PARTICLE_dt
				WRITE_PARTICLE_dt(outfile, ps[i]);
		#endif
		#endif
	}
	outfile.close();
}

/* 
//针对两种不同密度的流体
inline void write_fluid_particles_json(const char* filename1, const char* filename2, const SphBase& sphObj, int idx = 0)
{
	assert(idx >= 0 && idx < sphObj.getNumFluids());

	const std::vector<SphBase::FluidPart>& ps = sphObj.getFluidParticles(idx);
	int num = ps.size();
	real_t beta;
	std::vector<int> beta1_i;
	std::vector<int> beta2_i;
	for (int i = 0; i < num; ++i)
	{
		beta = ps[i].beta;
		if (beta == 1.0) {
			beta1_i.push_back(i);
		}
		else {
			beta2_i.push_back(i);
		}
	}

	std::ofstream outfile;
	outfile.open(filename1, ios::out);
	if (!outfile) { // open failed
		outfile.close();
		std::cout << "write_json(): open file failed!\n";
		return;
	}
	//根节点
	Json::Value root1;
	//子节点
	Json::Value position1;
	Json::Value velocity1;

	int i = 0;
	for (int index = 0; index < beta1_i.size(); ++index) {
		i = beta1_i[index];
		int num1 = beta1_i.size();
		//根节点属性
		root1["num"] = num1;
		position1["x"] = ps[i].position[0];
		position1["y"] = ps[i].position[2];
		position1["z"] = ps[i].position[1];
		root1["position"].append(position1);
		velocity1["x"] = ps[i].velocity[0];
		velocity1["y"] = ps[i].velocity[2];
		velocity1["z"] = ps[i].velocity[1];
		root1["velocity"].append(velocity1);
	}
	outfile << root1.toStyledString();
	outfile.close();

	outfile.open(filename2, ios::out);
	if (!outfile) { // open failed
		outfile.close();
		std::cout << "write_json(): open file failed!\n";
		return;
	}
	//根节点
	Json::Value root2;
	//子节点
	Json::Value position2;
	Json::Value velocity2;

	int i = 0;
	for (int index = 0; index < beta2_i.size(); ++index) {
		i = beta1_i[index];
		int num2 = beta2_i.size();
		//根节点属性
		root2["num"] = num2;
		position2["x"] = ps[i].position[0];
		position2["y"] = ps[i].position[2];
		position2["z"] = ps[i].position[1];
		root2["position"].append(position2);
		velocity2["x"] = ps[i].velocity[0];
		velocity2["y"] = ps[i].velocity[2];
		velocity2["z"] = ps[i].velocity[1];
		root2["velocity"].append(velocity2);
	}
	outfile << root2.toStyledString();
	outfile.close();
}
*/

inline void write_fluid_particles_json(const char* filename, const SphBase& sphObj, int idx = 0)
{
	assert(idx >= 0 && idx < sphObj.getNumFluids());

	const std::vector<SphBase::FluidPart>& ps = sphObj.getFluidParticles(idx);
	int num = ps.size();

	std::ofstream outfile;
	outfile.open(filename, ios::out);
	if (!outfile) { // open failed
		outfile.close();
		std::cout << "write_json(): open file failed!\n";
		return;
	}
	//根节点
	Json::Value root1;
	//子节点
	Json::Value position1;
	Json::Value velocity1;
	Json::Value beta1;
	//根节点属性
	root1["num"] = num;
	for (int i = 0; i < num; ++i) {
		root1["spacing_r"] = ps[i].d;
		position1["x"] = ps[i].position[0];
		position1["y"] = ps[i].position[2];
		position1["z"] = ps[i].position[1];
		root1["position"].append(position1);
		velocity1["x"] = ps[i].velocity[0];
		velocity1["y"] = ps[i].velocity[2];
		velocity1["z"] = ps[i].velocity[1];
		root1["velocity"].append(velocity1);
		beta1["value"] = ps[i].temp_beta;
		root1["temp_beta"].append(beta1);
	}
	outfile << root1.toStyledString();
	outfile.close();
}

//see 9
inline void write_candidate_particles(const char* filename, const SphBase& sphObj, int idx = 0) {
	assert(idx >= 0 && idx<sphObj.getNumCandidates());
	std::ofstream outfile;
	outfile.open(filename, std::fstream::binary);
	if (!outfile) { // open failed
		outfile.close();
		std::cout << "write_candidate_particles(): open file failed!\n";
		return;
	}
	// line 1, log file
	outfile << "for parameters and scene info, please refer to \""
		<< sphObj.getLogFileName() << "\"" << std::endl;
	// line 2, components writed
	outfile << "components writed: " << get_part_write_mask() << std::endl;
	// line 3, frame number & fluid index
	outfile << sphObj.getFrameNumber() << " (frame number), candidate index: " << idx << std::endl;
	// line 4, size
	const std::vector<SphBase::CandidatePart>& ps = sphObj.getCandidateParticles(idx);
	outfile << ps.size() << " (num particles)" << std::endl;
	for (size_t i = 0; i < ps.size(); ++i) {
	#ifdef WRITE_PARTICLE_position
		WRITE_PARTICLE_position(outfile, ps[i]);
	#endif
	}
	outfile.close();
}


//see 9
inline void write_solid_particles(const char* filename, const SphBase& sphObj, int idx = 0) {
	assert(idx >= 0 && idx<sphObj.getNumBounParts());
	std::ofstream outfile;
	outfile.open(filename, std::fstream::binary);
	if (!outfile) { // open failed
		outfile.close();
		std::cout << "write_solid_particles(): open file failed!\n";
		return;
	}
	// line 1, log file
	outfile << "for parameters and scene info, please refer to \""
		<< sphObj.getLogFileName() << "\"" << std::endl;
	// line 2, components writed
	outfile << "components writed: " << get_part_write_mask() << std::endl;
	// line 3, frame number & fluid index
	outfile << sphObj.getFrameNumber() << " (frame number), solid index: " << idx << std::endl;
	// line 4, size
	const std::vector<SphBase::BoundPart>& ps = sphObj.getBoundaryParticles(idx);
	outfile << ps.size() << " (num particles)" << std::endl;
	for (size_t i = 0; i < ps.size(); ++i) {
	#ifdef WRITE_PARTICLE_position
		WRITE_PARTICLE_position(outfile, ps[i]);
	#endif
	}
	outfile.close();
}

inline void read_fluid_particles(const char* filename, std::vector<real_t>& data, std::string* maskstr=0)
{
	std::ifstream infile;
	infile.open( filename, std::fstream::binary );
	if( !infile ) { // open failed
		infile.close();
		std::cout << "read_pos(): open file failed!\n";
		return;
	}

	long long num; char ssh1[505], ssh2[505], ssh3[505], ss[505];
	infile.getline(ssh1,500);
	infile.getline(ssh2,500);
	infile.getline(ssh3,500);
	infile >> num; infile.getline(ss,500);

	if(maskstr){
		*maskstr = std::string( ssh2 );
	}

	data.resize( num* get_part_size(std::string(ssh2))/sizeof(real_t) );
	infile.read( (char*)&data[0], data.size()*sizeof(real_t) );

	infile.close();
}

/*
* Draw wireframe of a rectangle
*/
inline void wireframe( glm::vec3 min, glm::vec3 max, const float color[4], float xyz_len=0, float width=1 )
{
	GLboolean light;
	light = glIsEnabled(GL_LIGHTING);
	if(GL_TRUE==light) glDisable(GL_LIGHTING);
	GLfloat line_w; glGetFloatv(GL_LINE_WIDTH, &line_w);
	glLineWidth(width);
	glColor4fv(color);
	glBegin(GL_LINE_LOOP);
	glVertex3f(min.x, min.y, min.z);
	glVertex3f(max.x, min.y, min.z);
	glVertex3f(max.x, max.y, min.z);
	glVertex3f(min.x, max.y, min.z);
	glEnd();
	glBegin(GL_LINE_LOOP);
	glVertex3f(min.x, min.y, max.z);
	glVertex3f(max.x, min.y, max.z);
	glVertex3f(max.x, max.y, max.z);
	glVertex3f(min.x, max.y, max.z);
	glEnd();
	glBegin(GL_LINES);
	glVertex3f(min.x, min.y, min.z);
	glVertex3f(min.x, min.y, max.z);
	glVertex3f(max.x, min.y, min.z);
	glVertex3f(max.x, min.y, max.z);
	glVertex3f(max.x, max.y, min.z);
	glVertex3f(max.x, max.y, max.z);
	glVertex3f(min.x, max.y, min.z);
	glVertex3f(min.x, max.y, max.z);
	glEnd();
	if(xyz_len>0.0) {
		float c[]={.0, .0, .0, 1.0};
		glBegin(GL_LINES);
		c[0]=1.0f; c[1]=.0f; c[2]=.0f;
		glColor3fv(c);
		glVertex3f(max.x, min.y, min.z);
		glVertex3f(max.x+xyz_len, min.y, min.z);
		c[0]=.0f; c[1]=1.0f; c[2]=.0f;
		glColor3fv(c);
		glVertex3f(min.x, max.y, min.z);
		glVertex3f(min.x, max.y+xyz_len, min.z);
		c[0]=.0f; c[1]=.0f; c[2]=1.0f;
		glColor3fv(c);
		glVertex3f(min.x, min.y, max.z);
		glVertex3f(min.x, min.y, max.z+xyz_len);
		glEnd();
	}
	if(GL_TRUE==light) glEnable(GL_LIGHTING);
	glLineWidth(line_w);
}

#endif // _FUTILITY_H_


