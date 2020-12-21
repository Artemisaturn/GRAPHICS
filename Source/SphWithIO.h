/* 
 * heliangliang (heliangliang.bj@gmail.com), USTB, 2014.05.14, CopyRight Reserved
 *
 * The SPH algrithm is implemented based on these papers:
 * 1. Smoothed particle hydrodynamics (Monaghan,J.J. 1992)			[Mon92]
 * 2. Smoothed particles: a new paradigm for animating highly deformable
 *	  bodies (1996)													[DC96]
 * 3. Weakly compressible SPH for free surface flows (2007)			[BT07]
 * 4. Predictive-Corrective Incompressible SPH (2009)				[SP09]
 * 5. Density contrast SPH interfaces (2008)						[SP08]
 * 6. Versatile Rigid-Fluid Coupling for Incompressible SPH (2012)	[AIS*12]
 * 7. Versatile surface tension and adhesion for SPH fluids (2013)	[AAIT13]
 * 8. SPH Fluids in Computer Graphics (2014)						[IOS*14]
 *
*/

#ifndef SPH_WITH_IO_H_
#define SPH_WITH_IO_H_


#include "SphWithBullet.h"
#include "gl_inc.h"


class SphWithIO : public SphWithBullet{

public:

	// callBack() will be called iteratively, using particle information
	// (e.g. pressure, density, velocity force etc.) to compute particle color
	void setFluidPartsColor( int fidx, void(*colorFunc1)(float reColor[4],const FluidPart& p), void(*colorFunc2)(float reColor[4], const FluidPart& p)){
		std::vector<FluidPart>& f_pts = m_Fluids[fidx].fluidParticles;
		for (int n = int(f_pts.size()), i = 0; i < n; ++i) {
			//colorFunc1(f_pts[i].color, f_pts[i]);
			
			if (f_pts[i].beta > 0.9) {
				colorFunc1(f_pts[i].color, f_pts[i]);
			}
			else {
				colorFunc2(f_pts[i].color, f_pts[i]);
			}
			
		}		
	}

	void setBoundPartsColor( int ridx, void(*colorFunc)(float reColor[4],const BoundPart& p) ){
		std::vector<BoundPart>& r_pts = m_Solids[ridx].boundaryParticles;
		for(int n=int(r_pts.size()),i=0;i<n;++i) colorFunc(r_pts[i].color,r_pts[i]);
	}

	//see 9
	void setCandidatePartsColor(int cidx, void(*colorFunc)(float reColor[4], const CandidatePart& p)) {
		std::vector<CandidatePart>& r_pts = m_Candidates[cidx].candidateParticles;
		for (int n = int(r_pts.size()), i = 0; i<n; ++i) colorFunc(r_pts[i].color, r_pts[i]);
	}

	//see 9
	void scanCandidateParticles(int fluidIdx, int candidateIdx);

	// draw fluid particles or boundary paricles of solids in OpenGL, iteratively
	// parameter test use to select particles, i.e., cut down some particles
	void oglDrawFluidParts(void(*draw1)(), void(*draw2)(), bool(*test)(const vec_t& p)=0) const;
	void oglDrawBounParts(void(*draw)(), bool(*test)(const vec_t& p)=0) const;
	void oglDrawCandidateParts(void(*draw)(), bool(*test)(const vec_t& p) = 0) const;//see 9
	void oglDrawSolid(int ridx) const;
	void oglDrawSolid(bool(*test)(int i)=0) const;


	// particles are sampled at regular cubic grid points
	// the cube is defined by parameter leftBottom and rightTop(Axis-Aligned box)
	// parameter fpart is the initial fluid paricle used to initialize fluid particles
	// parameter newFluid = true: add a new fluid, fluidIdx will be ignored
	// false: add particles to last fluid and the last three parameters will be ignored
	void addFluidCuboid( bool newFluid, int fluidIdx,
		const vec_t& leftBottom, const vec_t& rightTop, const FluidPart& fpart,
		real_t restDensity=1000, real_t viscosity=real_t(0.05), real_t tension=0 );
	// add particles to fluid[fluidIdx]

	void addFluidParts(int fluidIdx, const std::vector<FluidPart>& particles);

	//see 9
	void addEmptyFluid(real_t restDensity = 1000, real_t viscosity = real_t(0.05), real_t tension = 0);

	//see 9
	//void removeFluidParts(int fluidIdx, std::vector<FluidPart>::iterator itr);
	void removeFluidParts(int fluidIdx, int particle);

	//see 9
	void removeSolidObject(int solidIdx);

	//see 9
	void addCandidateFromPLY(const char* meshFileName, const char* sampleFileName, int fluidIdx, 
		const CandidatePart& cpart,real_t viscosity, const glm::mat4& transform, int mass, vec_t pos);

	//see 9
	void addIceCubid(int fluidIdx,
		const CandidatePart& cpart, real_t viscosity, const glm::mat4& transform, 
		real_t density, vec_t pos, bool dynamic, vec_t size);

	//see 11
	void addIceFromPLY(const char* meshFileName, const char* sampleFileName, int fluidIdx,
		const CandidatePart& cpart, real_t viscosity, const glm::mat4& transform, real_t density, vec_t pos, bool dynamic);

	//see 12
	void fluidTemperatureSet(int fluidIdx, real_t temperature);

	//see 12
	void candidateTemperatureSet(int candidateIdx, real_t temperature);

	//see 12
	void solidTemperatureSet(int solidIdx, real_t temperature);

	//see 9
	void candidateToSolid(int candidateIdx);

	//see 9
	void updateTrans();

	//see 9
	std::vector<vec_t> reSampling(std::vector<float>& vers);

	// the PLY file contain the sampled points
	void addFluidFromPLY( const char* filename,
		bool newFluid, int fluidIdx, const FluidPart& fpart,
		real_t restDensity=1000, real_t viscosity=real_t(0.05), real_t tension=0 );


	// if mass>0 dynamic=true
	void addRigidCuboid( const vec_t& halfWidth,
		const BoundPart& bpart, float mass, real_t viscosity,
		bool haveTop=true, const glm::mat4& transform=glm::mat4() );

	// convex hull of the mesh will be the collision shape, i.e., the mesh itself may not
	void addRigidFromPLY( const char* meshFileName, const char* sampleFileName,
		const BoundPart& bpart, float mass, real_t viscosity,
		const glm::mat4& transform=glm::mat4() );

	// "<convexname>0", "<convexname>1", "<convexname>2", ... are the convex parts of the mesh,
	// i.e., mainname is used to display in opengl, btCompoundShape of convecnum "<convexname>i"
	// is used for collision detection
	void addRigidFromPLY( const char* meshFileName, const char* sampleFileName,
		const char* convexMeshName, int convecnum,
		const BoundPart& bpart, float mass, real_t viscosity,
		const glm::mat4& transform=glm::mat4() );

	//see 9
	void addRigidFromPLYForCandidate(const char* meshFileName, const char* sampleFileName,
		const BoundPart& bpart, float mass, real_t viscosity,
		const glm::mat4& transform = glm::mat4());

	//see 9
	void addRigidFromPLYAndPosition(const char* meshFileName, const char* sampleFileName,
		const BoundPart& bpart, float mass, real_t viscosity, int candidateIdx, vec_t& pos,
		const glm::mat4& transform = glm::mat4());

	//see 9
	void solidToFluid(int solidIdx, int fluidIdx);

	//see 10
	void addFluidFromPLYAndResampling(int fluidIdx, const char* meshFileName, const char* sampleFileName, 
		const FluidPart& fpart, real_t viscosity, real_t restDensity, real_t tension, 
		const vec_t& pos);


	// mass must >0, dynamic is certainly true
	// non-tranformed rope is on x-axis
	void addSoftRopeLine(
		float halfWidth_x, int endFixedBit,
		const BoundPart& bpart, float mass, real_t viscosity,
		const glm::mat4& transform=glm::mat4(), int twoEndIdx[2]=0 );

	// non-tranformed cloth is on XoZ plane
	void addSoftClothPlane(
		float halfWidth_x, float halfWidth_z, float stretchDis_y, int cornerFixedBit,
		const BoundPart& bpart, float mass, real_t viscosity,
		const glm::mat4& transform=glm::mat4(), int fourCornerIdx[4]=0 );

	// tetrahedral mesh
	void addSoftVolumeCuboid(
		const vec_t& halfWidth,
		const BoundPart& bpart, float mass, real_t viscosity,
		const glm::mat4& transform=glm::mat4(), int eightCornerIdx[8]=0 );

	// each vetex is a node, edge a link
	void addSoftFromPly( const char* filename,
		const BoundPart& bpart, float mass, real_t viscosity,
		const glm::mat4& transform=glm::mat4() );


	// duplicate the last added solid
	void duplicateLastSolid( const glm::mat4& transform );


	void saveSolidMeshToPLY(
		int solidIdx, const char* filename, bool simulated, bool coordYupToZup=false) const;

	// non-transformed rigid mesh, and initial softbody mesh, coordYupToZup:true for blender
	void saveAllInitialSolidMeshToPLY(
		const char* namePrefix, bool coordYupToZup=false, bool staticSolidsTransformed=false) const;

	// trasnformed rigid mesh, and simulated softbody mesh, coordYupToZup:true for blender
	void saveAllSimulatedSolidMeshToPLY(
		const char* namePrefix, bool coordYupToZup=false, bool staticSolids=true) const;


	// Scene example
	//void setExampleScene(bool rigid=false);

protected:

	// particles are sampled at the surface of cube defined
	// by parameter leftBottom and rightTop(Axis-Aligned box)
	// parameter haveTop indicates sampling the top square of the cube or not
	void makeCubeSampleParts(
		std::vector<vec_t>& samples, const vec_t& halfWidth, bool haveTop, real_t r) const;
	void makeCubeMeshAndBtShape(
		GLTriMesh& mesh, btCollisionShape** shape, const vec_t& halfWidth, bool haveTop, real_t thickness) const;


}; // class SphWithIO


#endif // #ifndef SPH_WITH_IO_H_

