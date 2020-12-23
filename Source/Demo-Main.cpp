/**/
#define _SILENCE_STDEXT_HASH_DEPRECATION_WARNINGS 1
#include "gl_staff.h"
#include "bt_inc.h"
#include "vcg_inc.h"
#include <fstream>
#include <sstream>
#include "fUtility.h"
#include "wrap/ply/plylib.cpp"

#include "demo-mysph.h"
//#include "test.h"

mySPH sphDemo; // the global object of sph class


bool fluid_system_run = false, draw_pos = true, draw_boun = true;
bool boun_mode = true;
bool write_file = false;
bool draw_frame = true;
bool colorTrans = false;//see 12
bool draw_candidate = true;//see 9

#define PRINT_BOOL(a) std::cout << #a": " << a << "\n"

void key_r() { fluid_system_run = !fluid_system_run; PRINT_BOOL(fluid_system_run); }
void key_b() { draw_boun = !draw_boun; PRINT_BOOL(draw_boun); }
void key_f() { draw_pos = !draw_pos; PRINT_BOOL(draw_pos); }
void key_m() { boun_mode = !boun_mode; PRINT_BOOL(boun_mode); }
void key_w() { write_file = !write_file; PRINT_BOOL(write_file); }
void key_s() { draw_frame = !draw_frame; PRINT_BOOL(draw_frame); }
void key_c() { draw_candidate = !draw_candidate; PRINT_BOOL(draw_candidate); }//see 9
void key_p() { colorTrans = !colorTrans; PRINT_BOOL(draw_candidate); }//see 12

real_t phase1;
real_t phase2;
real_t phase3;
real_t phase4;

void colorFunc(float reColor[4], const mySPH::FluidPart& p) {

	reColor[3] = reColor[2] = reColor[1] = reColor[0] = 0;
	reColor[3] = 1;
	if (!colorTrans) {
		if (p.temperature > (273.15 + phase1) && p.temperature < (273.15 + phase2))
			reColor[2] = (p.temperature - (273.15 + phase1)) / -phase1;
		else if (p.temperature > (273.15 + phase2) && p.temperature < (273.15 + phase3)) {
			reColor[2] = 1 - (p.temperature - (273.15 + phase2)) / (phase3 - phase2) / 2;
			reColor[1] = (p.temperature - (273.15 + phase2)) / (phase3 - phase2);
		}
		else if (p.temperature > (273.15 + phase3) && p.temperature < (273.15 + phase4)) {
			reColor[2] = 0.5 - (p.temperature - (273.15 + phase3)) / (phase4 - phase3) / 2;
			reColor[1] = 1 - (p.temperature - (273.15 + phase3)) / (phase4 - phase3) / 2;
			reColor[0] = (p.temperature - 293.15) / (phase4 - phase3);
		}
		else {
			reColor[0] = 1;
			reColor[1] = 1;
			reColor[2] = 1;
		}
	}
	else {
		if (p.presure > (273.15 + phase1) && p.presure < (273.15 + phase2))
			reColor[2] = (p.presure - (273.15 + phase1)) / -phase1;
		else if (p.presure > (273.15 + phase2) && p.presure < (273.15 + phase3)) {
			reColor[2] = 1 - (p.presure - (273.15 + phase2)) / (phase3 - phase2) / 2;
			reColor[1] = (p.presure - (273.15 + phase2)) / (phase3 - phase2);
		}
		else if (p.presure > (273.15 + phase3) && p.presure < (273.15 + phase4)) {
			reColor[2] = 0.5 - (p.presure - (273.15 + phase3)) / (phase4 - phase3) / 2;
			reColor[1] = 1 - (p.presure - (273.15 + phase3)) / (phase4 - phase3) / 2;
			reColor[0] = (p.presure - 293.15) / (phase4 - phase3);
		}
		else {
			reColor[0] = 1;
			reColor[1] = 1;
			reColor[2] = 1;
		}
	}
}

void colorFunc_b(float reColor[4], const mySPH::BoundPart& p) {
	/*static const float r=sphDemo.getSpacing_r(), v_max = r*r*r * 1.8f, v_min = r*r*r * 1.0f;
	reColor[3] = 1; reColor[0]=reColor[1]=reColor[2]
		= std::max(0.0f, std::min(1.0f, 1-(p.volume-v_min)/(v_max-v_min)) );*/

	reColor[3] = reColor[2] = reColor[1] = reColor[0] = 0;
	reColor[3] = 1;
	if (p.temperature > (273.15 + phase1) && p.temperature < (273.15 + phase2))
		reColor[2] = (p.temperature - (273.15 + phase1)) / -phase1;
	else if (p.temperature > (273.15 + phase2) && p.temperature < (273.15 + phase3)) {
		reColor[2] = 1 - (p.temperature - (273.15 + phase2)) / (phase3 - phase2) / 2;
		reColor[1] = (p.temperature - (273.15 + phase2)) / (phase3 - phase2);
	}
	else if (p.temperature > (273.15 + phase3) && p.temperature < (273.15 + phase4)) {
		reColor[2] = 0.5 - (p.temperature - (273.15 + phase3)) / (phase4 - phase3) / 2;
		reColor[1] = 1 - (p.temperature - (273.15 + phase3)) / (phase4 - phase3) / 2;
		reColor[0] = (p.temperature - 293.15) / (phase4 - phase3);
	}
	else {
		reColor[0] = 1;
		reColor[1] = 1;
		reColor[2] = 1;
	}
}

//see 9
void colorFunc_c(float reColor[4], const mySPH::CandidatePart& p) {
	//reColor[3] = reColor[2] = 1;
	//reColor[0] = reColor[1] = 0.5f;//std::min(1.0f, 0.5f);

	reColor[3] = reColor[2] = reColor[1] = reColor[0] = 0;
	reColor[3] = 1;
	if (p.temperature > (273.15 + phase1) && p.temperature < (273.15 + phase2))
		reColor[2] = (p.temperature - (273.15 + phase1)) / -phase1;
	else if (p.temperature > (273.15 + phase2) && p.temperature < (273.15 + phase3)) {
		reColor[2] = 1 - (p.temperature - (273.15 + phase2)) / (phase3 - phase2) / 2;
		reColor[1] = (p.temperature - (273.15 + phase2)) / (phase3 - phase2);
	}
	else if (p.temperature > (273.15 + phase3) && p.temperature < (273.15 + phase4)) {
		reColor[2] = 0.5 - (p.temperature - (273.15 + phase3)) / (phase4 - phase3) / 2;
		reColor[1] = 1 - (p.temperature - (273.15 + phase3)) / (phase4 - phase3) / 2;
		reColor[0] = (p.temperature - 293.15) / (phase4 - phase3);
	}
	else {
		reColor[0] = 1;
		reColor[1] = 1;
		reColor[2] = 1;
	}
}

void colorFunc_1(float reColor[4], const mySPH::FluidPart& p) {

	reColor[3] = 1;
	reColor[2] = 1;
	reColor[1] = 1;
	reColor[0] = 1;
}

void colorFunc_2(float reColor[4], const mySPH::FluidPart& p) {

	reColor[3] = 1;
	reColor[2] = 0;
	reColor[1] = 0;
	reColor[0] = 1;
}

bool planecut(const vec_t& p) { if (p.x - p.z + 0.01f > 0) return true; else return false; }
void draw_unitsphere() { glCallList(1); }
void draw_unitsphere1() { glCallList(1); }
void draw_unitsphere2() { glCallList(2); }
bool rigid_test(int i) { return i == 0; }

static const int vedio_fps = 25;
static int vedio_next_framenum = 0;

void run_and_draw()
{
	if (fluid_system_run) {
		for (int i = 0; i < 1; ++i) {
			sphDemo.runOneStep();
			if (sphDemo.getSystemTime() >= vedio_next_framenum / float(vedio_fps)) break;
		}
	}

	glMatrixMode(GL_MODELVIEW);
	if(draw_pos){
		for (size_t i = 0; i < sphDemo.getNumFluids(); ++i) 
			sphDemo.setFluidPartsColor(int(i), colorFunc, colorFunc_2);
		sphDemo.oglDrawFluidParts(draw_unitsphere1, draw_unitsphere2/*, planecut*/);
	}

	//glMatrixMode(GL_MODELVIEW);
	//if (draw_pos) {
	//	sphDemo.setFluidPartsColor(0, colorFunc_1);
	///	sphDemo.setFluidPartsColor(1, colorFunc_2);
	//	sphDemo.oglDrawFluidParts(draw_unitsphere/*, planecut*/);
	//}

	if (draw_boun) {
		float c[4] = { 0,1,0.8f, 1 };
		glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, c);
		if (boun_mode) {
			//sphDemo.oglDrawSolid();
			for (size_t i = 0; i < sphDemo.getNumSolids(); ++i) {
				glStaff::hsl_to_rgb(float(i) / sphDemo.getNumSolids() * 330 + 30, 1, 0.5f, c);
				glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, c);
				sphDemo.oglDrawSolid(int(i));
			}
		}
		else {
			for (size_t i = 0; i < sphDemo.getNumSolids(); ++i)
				sphDemo.setBoundPartsColor(int(i), colorFunc_b);
			sphDemo.oglDrawBounParts(draw_unitsphere2);
		}
	}

	//see 9
	if (draw_candidate) {
		float c[4] = { 0,1,0.8f, 0 };
		glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, c);
		for (size_t i = 0; i < sphDemo.getNumCandidates(); ++i)
			sphDemo.setCandidatePartsColor(int(i), colorFunc_c);
		sphDemo.oglDrawCandidateParts(draw_unitsphere);
	}
	if (colorTrans) {
		phase1 = -300;
		phase2 = 10000;
		phase3 = 20000;
		phase4 = 30000;
	}
	else {
		phase1 = -30;  //-30 
		phase2 = 0;   //0 
		phase3 = 25;//25;
		phase4 = 50; //50;
	}
	if (draw_frame) {
		glm::vec3 s_min(sphDemo.getSpaceMin()[0], sphDemo.getSpaceMin()[1], sphDemo.getSpaceMin()[2]);
		glm::vec3 s_max(sphDemo.getSpaceMax()[0], sphDemo.getSpaceMax()[1], sphDemo.getSpaceMax()[2]);
		float c[] = { 1,1,1, 1 };
		wireframe(s_min, s_max, c, 1);
	}

	char ss[50];
	sprintf(ss, "sys time: %f", sphDemo.getSystemTime());
	glStaff::text_upperLeft(ss, 1);
	sprintf(ss, "  dt(ms): %f", 1000 * sphDemo.getDt());
	glStaff::text_upperLeft(ss);
	sprintf(ss, "   frame: %d", sphDemo.getFrameNumber());
	glStaff::text_upperLeft(ss);

#ifdef WC_TIMEADAPTIVE
	sprintf(ss, "active %%: %.2f", 100 * sphDemo.wc_percentOfActive);
	glStaff::text_upperLeft(ss);
#endif

	if (write_file && fluid_system_run &&
		sphDemo.getSystemTime() >= vedio_next_framenum / float(vedio_fps)) {
		wchar_t ss[50];
		swprintf(ss, L"img/%d.png", vedio_next_framenum);
		int w, h; glStaff::get_frame_size(&w, &h);
		il_saveImgWin(ss, 0, 0, w, h); // png

		char t[50];
		for (int i = 0; i < sphDemo.getNumFluids(); i++) {
			//beta=1 and beta=0.1 
			sprintf(t, "pos/%d_%d_f.pos", vedio_next_framenum, i);
			write_fluid_particles(t, sphDemo, i); // pos
		}

		char ct[50];
		for (int i = 0; i < sphDemo.getNumCandidates(); i++) {
			sprintf(ct, "pos/%d_%d_c.pos", vedio_next_framenum, i);
			write_candidate_particles(ct, sphDemo, i); // pos
		}

		char dt[50];
		for (int i = 0; i < sphDemo.getNumSolids(); i++) {
			sprintf(dt, "pos/%d_%d_d.pos", vedio_next_framenum, i);
			write_solid_particles(dt, sphDemo, i); // pos
		}

		char json[50];
		for (int i = 0; i < sphDemo.getNumFluids(); i++) {
			sprintf(json, "json/%d_%d_f.json", vedio_next_framenum, i);
			write_fluid_particles_json(json, sphDemo, i);
		}

		glm::mat4 cube_trans;
		for (int i = 0; i < sphDemo.getNumSolids(); i++) {
			sprintf(t, "mat4/%d_%d.mat4", vedio_next_framenum, i); // mat4
			sphDemo.getRigidBodyTransform(i, cube_trans);
			glStaff::save_mat_to_file(t, cube_trans);
		}

		sprintf(t, "solid-ply/%d", vedio_next_framenum);
		sphDemo.saveAllSimulatedSolidMeshToPLY(t, true, false);

		++vedio_next_framenum;
		if (sphDemo.getSystemTime() > 30.1f) exit(0);
	}
	else {
		if (sphDemo.getSystemTime() == 0)
			vedio_next_framenum = 0;
		else
			vedio_next_framenum = int(sphDemo.getSystemTime() * vedio_fps + 0.5f);
	}
}

void draw(const glm::mat4& mat_model, const glm::mat4& mat_view)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW); glLoadMatrixf(&mat_view[0][0]);
	// world
	glMultMatrixf(&mat_model[0][0]);
	//vec_t sp = (sphDemo.getSpaceMax()+sphDemo.getSpaceMin())/2;
	//glTranslatef(-sp.x, 0, -sp.z);
	run_and_draw();
}

int main(int argc, char** argv)
{
	sphDemo.setupScene();
	sphDemo.saveAllInitialSolidMeshToPLY("solid-ply/initial", true, true);
	/*
	for(size_t i=0; i<100000000L; ++i);
	for(int i=0; i<3; ++i){
	mySPH* sph = new mySPH();
	sph->runOneStep();
	sph->runOneStep();
	delete sph;
	}//*/

#ifdef PREDICTION_PCISPH
	glStaff::init_win(1000, 800, "Demo PCISPH", "C:\\Windows\\Fonts\\lucon.ttf");
#else
	glStaff::init_win(1000, 800, "Demo WCSPH", "C:\\Windows\\Fonts\\lucon.ttf");
#endif

	glStaff::init_gl();

	GLfloat vec4f[] = { 1, 1, 1, 1 };
	glLightfv(GL_LIGHT0, GL_DIFFUSE, vec4f); // white DIFFUSE, SPECULAR
	glLightfv(GL_LIGHT0, GL_SPECULAR, vec4f);

	glStaff::set_mat_view(glm::lookAt(glm::vec3(10, 15, 15), glm::vec3(0, 3, 0), glm::vec3(0, 1, 0)));

	glm::mat4 mat_view;
	if (glStaff::load_mat_from_file("matrix_view", mat_view))
		glStaff::set_mat_view(mat_view);

	glStaff::add_key_callback('R', key_r, L"run sph");
	glStaff::add_key_callback('B', key_b, L"boundary particle");
	glStaff::add_key_callback('F', key_f, L"fluid particle");
	glStaff::add_key_callback('M', key_m, L"boundary mode");
	glStaff::add_key_callback('W', key_w, L"save pos continuously every video frame");
	glStaff::add_key_callback('S', key_s, L"draw frame of spacemin and spacemax");
	glStaff::add_key_callback('C', key_c, L"candidate particle");//see 9
	glStaff::add_key_callback('P', key_p, L"color trans");//see 12

	glNewList(1, GL_COMPILE);
	//glutSolidSphere(sphDemo.getSpacing_r() / 2, 10, 10);
	glutSolidSphere(0.15 / 2, 10, 10);
	glEndList();
	glNewList(2, GL_COMPILE);
	glutSolidSphere(0.3 / 2, 10, 10);
	glEndList();


	glStaff::renderLoop(draw);

	return 0;
}

#if defined(_MSC_VER) && defined(_DEBUG)
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
class _LeakCheckStatic {
public:
	_LeakCheckStatic() {
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
