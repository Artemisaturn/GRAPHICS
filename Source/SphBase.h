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
 * 9. Cornelis, Jens, et al. "Liquid boundaries for implicit incompressible SPH." Computers & Graphics 52 (2015): 72-78.
*/
#define _SILENCE_STDEXT_HASH_DEPRECATION_WARNINGS 1//see 9
#ifndef SPH_BASE_H_
#define SPH_BASE_H_


#include "Vec23.h"
#include <vector>
#include <set>
#include <string>
#include <algorithm>
#include <fstream>
#include <cassert>
#include <omp.h>
#include "stdafx.h"//see 9

//#define PREDICTION_PCISPH // define this at project's property
#define NEGLECT_NEGATIVE_PRESSURE
#define DIMENSION 3

std::string get_date_time_string(bool year=true, bool time=true);

// typedef of basic float type and vector type
// change these typedef to use float or double, and vec3 or vec2
// NOTE: for bullet and OpenGL, we use float in spite of what real_t is
typedef float real_t;
typedef vec_dim<DIMENSION,real_t>::T vec_t;
typedef vec_dim<3,real_t>::T vec3_t;

#ifdef _WIN64
typedef long long int_t;
#else
typedef int int_t;
#endif
typedef vec_dim<vec_t::dim,int_t>::T veci_t; // corresponds to vec_t's dimention

/*
 * ----- fluid and solid --------------------------------------------------------------------------
*/

// basic particle class
class Particle {
public:
    vec_t position;	// position
    vec_t velocity;	// velocity
	vec_t vel_adv; //for iisph
	vec_t acce_adv;//for iisph
	vec_t acce_presure;
	vec_t dii;//for iisph
	vec_t sum_dijpj;//for iisph
	vec_t n;
	real_t aii;//for iisph
	real_t rho_adv;//for iisph
	real_t p_l;//for iisph
    real_t volume;
    real_t vol_adv;
	real_t temperature;//see 12
	real_t temperatureNext;//see 12
    float color[4];	// RGBA color
    real_t d; //dia
};

// fluid particle
class FluidParticle : public Particle {
public:
    real_t density;		// density of particle
    real_t density_adv;
    real_t presure;		// WCSPH or PCISPH
    vec_t acceleration;	// acceleration
    real_t tempvolume;
    real_t fm0;
    real_t rho0;
    real_t beta;
    real_t temp_beta;

    real_t DFalpha;
    vec_t DFalpha1;
    real_t DFalpha2;
    real_t DFkappa;
    real_t DFkappaV;
    real_t DFdivergenceDeviation;

    real_t VFvolume;
    real_t VFvolume_adv;
    real_t VFalpha;
    vec_t VFalpha1;
    real_t VFalpha2;
    real_t VFalpha3;
    real_t VFkappa;
    real_t VFkappaV;
    real_t VFdivergenceDeviation;

#ifdef WC_TIMEADAPTIVE
	real_t t_last, dt_raw, dt, drho;
	vec_t pos_last;
	char sync_factor, ns_factor;
	bool active, dirty;
#else
#ifdef WC_ADT
	real_t dt;
	real_t h;
	real_t r;
	real_t bt;
#endif
#endif

#ifdef II_TIMEADAPTIVE
	real_t t_last, dt_raw, dt, drho;
	vec_t pos_last;
	char sync_factor, ns_factor;
	bool active, dirty;
#else
#ifdef II_ADT
	real_t dt;
	real_t h;
	real_t r;
	real_t bt;
#endif
#endif

#ifdef PREDICTION_PCISPH
    vec_t acce_pressure;
    vec_t pos_prediction;
#endif

};

// boundary particle, see [Pap2012]
class BoundaryParticle : public Particle {
public:
    real_t volume; // [AIS*12] equation (4), use kerSumFromSelf
    // sum of kernel only including neighbours from rigid self
    // used to accelerate the coputation of volume for rigids
	// for softbody, kerSumFromSelf=0
    real_t kerSumFromSelf;
    vec_t force;
};

//see 9
class CandidateParticle : public Particle {
public:
	vec_t targetPosition;
	bool ableToTrans = false;
	real_t volume;
	real_t density;
	real_t mass;
	vec_t force;
	//vec_t angularForce;//see 11
	vec_t linearForce;
	vec_t xab;
	vec_t torque;
	real_t latentHeat = 335000;
};

class SphObject{
public:
    //SphObject() : mask(true) { }
    //bool mask; // if mask=false, this object will not be processed
    std::string name;

};

//see 9
class SphCandidate : public SphObject {
public:
	typedef CandidateParticle Part;

	SphCandidate(real_t viscosity, const char* meshFileName, const char* sampleFileName, real_t mass, int fluidIdx, glm::mat4 trans, vec_t pos) :
	viscosity_alpha(viscosity), meshFileName(meshFileName), sampleFileName(sampleFileName), fluidIdx(fluidIdx), mass(mass), trans(trans), pos(pos){}

	SphCandidate(real_t viscosity, real_t density, int fluidIdx, glm::mat4 trans, vec_t pos) :
		viscosity_alpha(viscosity), fluidIdx(fluidIdx), density(density), trans(trans), pos(pos) {}

	real_t viscosity_alpha;
	real_t density;
	vec_t centerOfMass;
	vec_t totalTorque;
	vec_t totalForce = vec_t(0,0,0);
	vec_t gForce;
	vec_t angularVelocity;
	vec_t linearVelocity;
	vec_t linearForce;
	real_t mass;
	real_t singleMass;
	real_t volume;
	vec_t inertia;
	glm::mat3 inertiaTensor;
	real_t meltPoint = 0+273.15;
	real_t heatCapacity = 21000;
	int tansNum = 0;
	int correspondingSolidIdx;

	const char* meshFileName;
	const char* sampleFileName;
	int fluidIdx;
	//bool interaction = false;//see 11
	bool dynamic;
	vec_t angularMomentum;
	glm::vec3 vec3angularMomentum;
	glm::vec3 vec3angularVelocity;
	glm::mat4 trans;
	vec_t pos;

	std::vector<Part> candidateParticles;
	std::vector<vec_t> innerPositions;

	void scanTransbility(std::vector<Part>& candidateParticles);
	void particleToSolid(Part& candidatePart, int solidIdx);
	void doneTransformingToSolidObject(int solidIdx);

};

//see 11

// SPH fluid
class SphFluid : public SphObject {
public:
    // change these typedefs to implement other data structure
    typedef FluidParticle Part; // fluid particle type, e.g. FluidParticle

public:
    SphFluid(real_t restDensity, real_t viscosity, real_t tension) :
      restDensity_rho0(restDensity), viscosity_alpha(viscosity), surfaceTension_gamma(tension) {}
    
    //real_t particleMass_m0;	 // mass of one fluid particle
    real_t restDensity_rho0;	 // rho0, water is 1000.0
    real_t viscosity_alpha;		 // viscosity constant in fluid
    real_t surfaceTension_gamma; // surface tension constant
	bool isDynamic = true;//see 11

    std::vector<Part> fluidParticles;

    // Remove particles that are outside of the simulation space defined by spaceMin and spaceMax
    //  return number of particles removed
    int removeOutsideParts(const vec_t& spMin, const vec_t& spMax);

    // For particles that are outside of the simulation space defined by spaceMin and spaceMax,
    // correct their position and velocity, as if spaceMin and spaceMax defined 6 walls
    // return number of particles corrected
    void correctOutsideParts(const vec_t& spMin, const vec_t& spMax, real_t v0);

    // For particles that are outside of the simulation space defined by spaceMin and spaceMax,
    // change spaceMin and spaceMax to contain them
    // return true if spaceMin or spaceMax is changed
    bool adjustSpace(vec_t& spMin, vec_t& spMax) const;

};


class MBtSolid; // statement, BtSolid is defined in SphWithBullet.h

// SPH solid, e.g., rope, cloth, softbody, rigid
class SphSolid : public SphObject  {
public:
    // change these typedefs to implement other data structure
    typedef BoundaryParticle Part; // solid particle type, e.g. BoundaryParticle

	enum SolidType{
		RIGIDBODY,
		SOFTBODY,
	};

public:
    SphSolid( bool dynamic, SolidType type, real_t viscosity, MBtSolid* btObj=0 )
		: dynamic(dynamic), type(type), viscosity_alpha(viscosity), mbtSolid_ptr(btObj) 
	{ if(!dynamic) type=RIGIDBODY; }
    
    bool dynamic; // true: dynamic solid, false: static rigid
	SolidType type; // type of this solid
    real_t viscosity_alpha;
    std::vector<Part> boundaryParticles;
	// for rigid: the psitions of boundaryParticles when transform is Identity
	// for softbody: the initial positions of nodes of btSoftBody
    std::vector<vec_t> innerPositions;

	MBtSolid* mbtSolid_ptr; // pointer to solid object (btRigidBody, btSoftBody, ...)

protected:
	// update boundaryParticles follow the solid updated by bullet
    // for rigid, recompute boundaryParticles from innerPositions, i.e., using rigid's transform
    // pragmeter tranform is OpenGL transform matrix (tranform[0-3] is the first column)
    void updatePaticles(
        const vec_t& centerOfMass, const float* tranform,
        const vec_t& linearVelocity, const vec3_t& angularVelocity);
public:
	// for rigidbody, use rigid's transform
	// for softbody, use bullet btSoftBody::tNodeArray
	void updatePaticles(); // implement int SphWithBullet.cpp

    // for rigid, compute total force and torque from boundaryParticles' forces
    void rigidForceJoin(const vec_t& centerOfMass, vec_t& totalForce, vec3_t& totalTorque) const;

    // For particles that are outside of the simulation space defined by spaceMin and spaceMax,
    // change spaceMin and spaceMax to contain them
    // return true if spaceMin or spaceMax is changed
    bool adjustSpace(vec_t& spMin, vec_t& spMax) const;

	// get the index of four corners of softbody (cloth)
	//void getFourCornerIdxOfCloth(int idx[4]);
	// get the index of two ends of softbody (rope)
	//void getTwoEndIdxOfRope(int idx[2]) {idx[0]=0; idx[1]=int(innerPositions.size())-1;}

};



/*
 * 2D or 3D SPH simulator base class --------------------------------------------------------------
 * base data structure and neighbour search using uniform grid
*/
class SphBase{

public:
    // change these typedefs to implement other data structure
    typedef SphFluid Fluid;		// sph fluid type, e.g. SphFluid
    typedef SphSolid Solid;		// sph solid type, e.g. SphSolid
	typedef SphCandidate Candidate; //see 9
    typedef Fluid::Part FluidPart;	// e.g. FluidParticle
    typedef Solid::Part BoundPart;	// e.g. BoundaryParticle
	typedef Candidate::Part CandidatePart; //see 9

public:
    // Constructor, Destructor
    // redirect buffer of m_clog to log file "SphBase.log"
    // so, one can use m_clog to record something
    SphBase();
    virtual ~SphBase();

    // setup the sph fluid scene, e.g. add fluids and solids, set some parameters
    virtual void setupScene() = 0;

    // run one step of sph, i.e. event,advance,++frameNumber,++systemTime
    void runOneStep();

	std::string getLogFileName() const { return m_logFileName; };

	std::ostream m_clog;

private:
	std::ofstream m_logFile;
	std::string m_logFileName;

protected:

    // do something every time step, e.g. water outlet, work with runOneStep()
    virtual void stepEvent() = 0;

    // the computation of one step of sph advance, implement in subclasses
    virtual void sphStep() = 0;

    // time integration, Euler-Cromer, [IOS*14]
    void updateFluids();

	//see 9
	void updateCandidates();

	//see 11
	void calculateCenterOfMass(int candidateIdx);

	//see 11
	void calculateInertia(int candidateIdx);

	//see 11
	void calculateInertiaTensor(int candidateIdx);
	//see 11
	void applyMotionForCandidates(int candidateIdx, glm::vec3& rot, glm::vec3& trans);

	//see 12
	void updateTemperature();

	//see 12
	void temperatureTransmitFromFluidToCandidate(FluidPart& fp, CandidatePart& cp, real_t dis, real_t density, 
		real_t thermalDiffusionConstant);

	//see 12
	void temperatureTransmitFromCandidateToFluid(CandidatePart& cp, FluidPart& fp, real_t dis, real_t density,
		real_t thermalDiffusionConstant);

	//see 12
	void temperatureTransmitFromSolidToCandidate(BoundPart& bp, CandidatePart& cp, real_t dis, real_t density,
		real_t thermalDiffusionConstant);

	//see 12
	void temperatureTransmitFromCandidateToSolid(CandidatePart& cp, BoundPart& bp, real_t dis, real_t density,
		real_t thermalDiffusionConstant);

	//see 12
	void temperatureTransmitFromFluidToSolid(FluidPart& fp, BoundPart& bp, real_t dis, real_t density,
		real_t thermalDiffusionConstant);

	//see 12
	void temperatureTransmitFromSolidToFluid(BoundPart& bp, FluidPart& fp, real_t dis, real_t density,
		real_t thermalDiffusionConstant);

	//see 12
	void temperatureTransmitFromSolidToSolid(BoundPart& bp1, BoundPart& bp2, real_t dis, real_t density,
		real_t thermalDiffusionConstant);

	//see 12
	void temperatureTransmitFromFluidToFluid(FluidPart& fp1, FluidPart& fp2, real_t dis, real_t density,
		real_t thermalDiffusionConstant);

	//see 12
	void temperatureTransmitFromCandidateToCandidate(CandidatePart& cp1, CandidatePart& cp2, real_t dis, 
		real_t density, real_t thermalDiffusionConstant);

	//see 12
	void temperatureTransmitFromAirToCandidate(CandidatePart& cp, real_t airTemperature,
		real_t space, real_t heatCapacity = 2100, real_t thermalConductivity = 0.001);//model of ice

	//see 12
	void addFluidPartVector(int fluidIdx, const std::vector<FluidPart> particles);

    // compute BoundaryPart::kerSumFromSelf of earch boundary particle
    void prepareSolidPartWeight();

    // compute volume of earch boundary particle
    //  i.e. using BoundaryPart::kerSumFromSelf to accelerate the computing
    void updateSolidPartWeight();

	// compute hash code of the executing of the program, used for comparison
	int computeHashCodeOfPos();

	// print information of the scene
	virtual void logInfoOfScene();


public:

    // pubilic interface get(), set(), add()
    real_t	getSpacing_r() const { return m_TH.spacing_r; }
    real_t	getSmoothRadius_h() const { return m_TH.smoothRadius_h; }
    vec_t	getSpaceMin() const { return m_TH.spaceMin; }
    vec_t	getSpaceMax() const { return m_TH.spaceMax; }

    real_t	getDt() const { return m_TH.dt; }
    vec_t	getGravity_g() const { return m_TH.gravity_g; }
    real_t	getDensityErro_eta() const { return m_TH.densityErro_eta; }

    int		getFrameNumber() const { return m_TH.frameNumber; }
    real_t	getSystemTime() const { return m_TH.systemTime; }

    int		getErrorCount() const { return m_EC.zeroDis+m_EC.tooManyNeigb; }
    int		getErrorCountZeroDis() const { return m_EC.zeroDis; }
    int		getErrorCountManyNeigb() const { return m_EC.tooManyNeigb; }
    int		getHashCodeOfPos() const { return m_EC.hashCodeOfPos; }

    int getNumFluids() const { return int(m_Fluids.size()); }
    int getNumSolids() const { return int(m_Solids.size()); }
	int getNumCandidates() const { return int(m_Candidates.size()); }//see 9
    const std::vector<FluidPart>& getFluidParticles(int i) const
        { return m_Fluids[i].fluidParticles; }
    const std::vector<BoundPart>& getBoundaryParticles(int i) const
        { return m_Solids[i].boundaryParticles; }
	const std::vector<CandidatePart>& getCandidateParticles(int i) const//see 9
		{return m_Candidates[i].candidateParticles;}
    // return the sph fluid added
    Fluid& addFluid(Fluid& f) { m_Fluids.push_back(f); return m_Fluids.back(); }
    //Solid& addSolid(Solid& r) { m_Solids.push_back(r); return m_Solids.back(); }

    int getNumParts() const { return getNumFluidParts() + getNumBounParts() + getNumCandidateParts(); }
    int	getNumFluidParts() const{
        int re = 0;
        for(size_t num=m_Fluids.size(),i=0; i<num; ++i)
            re += int(m_Fluids[i].fluidParticles.size());
        return re;
    }
    int	getNumBounParts() const{
        int re = 0;
        for(size_t num=m_Solids.size(),i=0; i<num; ++i)
            re += int(m_Solids[i].boundaryParticles.size());
        return re;
    }
    int	getNumCandidateParts() const {
        int re = 0;
        for (size_t num = m_Candidates.size(), i = 0; i < num; ++i)
            re += int(m_Candidates[i].candidateParticles.size());
        return re;
    }

protected:

    // thresholds
    class ThresholdsBase{
    public:
        real_t spacing_r;		// spacing of particles, m0 = r^3 * rho0
        real_t smoothRadius_h;	// smooth kernel core radius, h = 2*r, ~32 neighbors
		real_t airTemperature;  //see 12
		real_t Kelvin = 273.15; //see 12
		bool applyCohesion = true;//see 12
		bool applyAdhesion = true;//see 12
		bool phaseTransition = true;//see 12
		bool applyDensityFluctuation = false;//see 12
        bool enable_energy_computation = false;
        int frequency_timer = 0;
        real_t energy_tracing_frequency = 0.5;
        vec_t spaceMin;			// min point of simulate space
        vec_t spaceMax;			// max point of simulate space

        real_t dt;				// time step length, delta t
        vec_t gravity_g;		// acceleration of gravity
        real_t densityErro_eta;	// the maximally allowed density fluictuation from rho0
        real_t soundSpeed_cs;   // speed of the sound in the fluid
		real_t h;
		real_t r;
		real_t bt;
		int count = 0;
		real_t simulationTime = 20;

        int frameNumber;		// the number of steps have run
        real_t systemTime;		// the time of simulation

        bool adjustDensity = false;
        real_t boundaryDensity = 1000;

        int dF_iteration;  // the number of divergenceFree iteration
        int cV_iteration; // the number of constantVolume or constantDensity iteration
        real_t potential_energy; // potential energy
        real_t kinetic_energy; //kinetic energy
        real_t sum_energy;
    } m_TH;

    // Error count
    class ErrorCount{
    public:
        ErrorCount():zeroDis(0),tooManyNeigb(0),hashCodeOfPos(0) { }
        int zeroDis;		// the number of dis of neighbours is 0
        int tooManyNeigb;	// the num of neighbours exceed max value
        int hashCodeOfPos;
    } m_EC;

    
    // particle(fluid or solid) index
    class PartIdx{
    public:
        PartIdx() { }
        PartIdx(int _t,int _i) : t(_t), i(_i) { }
        inline bool operator<(const PartIdx& b) const
            { return t<b.t || (t==b.t && i<b.i); }
        inline bool operator==(const PartIdx& b) const
            { return t==b.t && i==b.i; }
        int t;	// <0: m_Fluids[-tpye-1], >0: m_Solids[tpye-1], =0: NULL ptr
        int i;	// index of m_Fluids[-tpye-1].fluidParticle or m_Solids[tpye-1].boundaryParticles

        inline bool valid() const { return t!=0 /*&& i>=0*/; }
        inline bool isFluid() const { return t<0;  }
		inline bool isSolid() const { return (t<21&&t>0); }// see 11
		inline bool isCandidate() const { return (t>20); }// see 11
        inline int toFluidI() const { return -t-1; }
        inline int toSolidI() const { return t-1; }
		inline int toCandidateI() const { return t - 21; }//see 11
        inline static int tFromFluidI(int k) { return -(k+1); }
        inline static int tFromSolidI(int k) { return (k+1); }
		inline static int tFromCandidateI(int k) { return (k + 21); }//see 11
    };


    // for earch neighbour, save its PartIdx and distance to master particle
    class Neigb{
    public:
        Neigb() { }
        Neigb(PartIdx& idx, real_t d) : pidx(idx), dis(d) { }
        PartIdx pidx; real_t dis;
    };
    // structure for saving neighbours and constructing space grid
    class NeigbStr{
    public:
        static const int maxNeigbNum = 48;
        Neigb neigs[maxNeigbNum]; int num;
        void pushBack(const Neigb& a)
            { if(num<maxNeigbNum){neigs[num]=a; ++num;}
              /*else ++m_EC.tooManyNeigb;*/ }
        void pushBack(const PartIdx& idx, const real_t& dis)
            { if(num<maxNeigbNum){Neigb& ref=neigs[num]; ref.pidx=idx; ref.dis=dis; ++num;}
              /*else ++m_EC.tooManyNeigb;*/ }
        void pushBack(const PartIdx& idx, const real_t& dis, int& errorCount)
            { if(num<maxNeigbNum){Neigb& ref=neigs[num]; ref.pidx=idx; ref.dis=dis; ++num;}
              else ++errorCount; }
        void pushBack(const int& t, const int& i, const real_t& dis)
            { if(num<maxNeigbNum)
                {Neigb& ref=neigs[num]; ref.pidx.t=t; ref.pidx.i=i; ref.dis=dis; ++num;}
              /*else ++m_EC.tooManyNeigb;*/ }
        void clear() { num=0; }
    };


    // data menbers, fluids and solids
    std::vector<Fluid>	m_Fluids;
    std::vector<Solid>	m_Solids;
	std::vector<Candidate> m_Candidates; // see 9

    // the neighbour of a fluid particle and of a solid particle
    std::vector<std::vector<NeigbStr>> mg_NeigbOfFluids;
    std::vector<std::vector<NeigbStr>> mg_NeigbOfSolids;
	std::vector<std::vector<NeigbStr>> mg_NeigbOfCandidates;//see 11


    // get fluid or boundary particle from PartIdx
    const FluidPart& getFluidPartOfIdx(const PartIdx& idx) const
        { return m_Fluids[idx.toFluidI()].fluidParticles[idx.i]; }
    const BoundPart& getBoundPartOfIdx(const PartIdx& idx) const
        { return m_Solids[idx.toSolidI()].boundaryParticles[idx.i]; }
	//see 11
	const CandidatePart& getCandidatePartOfIdx(const PartIdx& idx) const
		{return m_Candidates[idx.toCandidateI()].candidateParticles[idx.i];}


    // find neighbours of each paritcle, parameters are for SphWithBullet::prepareSolidPartWeight
	// parameter RR_same: for rigid, search neighbor of itself or not (use kerSumFromSelf)
	// for softbody solid we have to search neighbor of itself
    void neighbourSearch(bool fluidPart=true, bool solidPart=true, bool candidatePart = true, bool RR_same=false);

	static const int WC_MULTITHREADS_NUM = 160;
	#define PARALLEL_SCHEDULE schedule( dynamic, std::max(100, getNumFluidParts()/WC_MULTITHREADS_NUM) )

// ------------------------------- neighbour search -----------------------------------------------

private:	// these members are about implementation details of neighbourSearch()

    class SpaceGridStruct{
    public:
		SpaceGridStruct() : num(0), min(0), divisor(0), factor(0),
			lastSpaceMin(1), lastSpaceMax(-1), lastSmoothRadius_h(-1) { }
	
		vec_t lastSpaceMin, lastSpaceMax;
		real_t lastSmoothRadius_h;
	
        std::vector<PartIdx> gridFirst;	// index of the first particle in a grid [i,j,k]
        veci_t	num;			// the number of cell in each dimention, eg. x,y,...
        vec_t	min;			// min coordinate x,y,... of the grid space
        real_t	divisor;		// the factor to calculate the coordinate x,y,...
        veci_t	factor;		// ( 1, Num.x, Num.x*Num.y, ... )
        std::vector<int_t> offset;	// the 5*5*... neighbors' offset of the center
        std::vector<std::vector<PartIdx>> nextOfFluids;
        std::vector<std::vector<PartIdx>> nextOfSolids;
		std::vector<std::vector<PartIdx>> nextOfCandidates;// see 11
        // get grid index from postion
        int_t gridOfPos(const vec_t& pos) const
            { return ((pos-min)*divisor).to<veci_t::type>().dot(factor); }
        // initialize the struture of space grid
        void initialize(const vec_t& spaceMin, const vec_t& spaceMax, real_t radius_h);

    } mg_NS; // Neighbour Search structure

    // insert particles into the space grid
    void gridSetup();

	//old one
    /*const PartIdx& getNextPIdx(const PartIdx& idx) const
        { return idx.isFluid()
                ? mg_NS.nextOfFluids[idx.toFluidI()][idx.i]
                : mg_NS.nextOfSolids[idx.toSolidI()][idx.i]; }*/
	//see 11 one
	const PartIdx& getNextPIdx(const PartIdx& idx) const
	{
		if (idx.isFluid()) {
			return mg_NS.nextOfFluids[idx.toFluidI()][idx.i];
		}
		if (idx.isSolid()) {
			return mg_NS.nextOfSolids[idx.toSolidI()][idx.i];
		}
		if (idx.isCandidate()) {
			return mg_NS.nextOfCandidates[idx.toCandidateI()][idx.i];
		}
	}

	//old one
    /*const vec_t& getPosOfPIdx(const PartIdx& idx) const
        { return idx.isFluid()
                ? m_Fluids[idx.toFluidI()].fluidParticles[idx.i].position
                : m_Solids[idx.toSolidI()].boundaryParticles[idx.i].position; }*/

	//see 11
	const vec_t& getPosOfPIdx(const PartIdx& idx) const
	{
		if (idx.isFluid()) {
			return m_Fluids[idx.toFluidI()].fluidParticles[idx.i].position;
		}
		if (idx.isSolid()) {
			return m_Solids[idx.toSolidI()].boundaryParticles[idx.i].position;
		}
		if (idx.isCandidate()) {
			return m_Candidates[idx.toCandidateI()].candidateParticles[idx.i].position;
		}
	}

protected:

	inline real_t ker_W(real_t r) const
		{ return ker_spline(r); /*return ker_spiky(r);*/ }
	inline real_t ker_W_grad(real_t r) const
		{ return ker_spline_grad(r); /*return ker_spiky_grad(r);*/ }

	//see 12
	inline real_t ker_W_laplacian(real_t r) const
	{
		return ker_laplacian_vis(r); }

private:

	// update internal numbers used to calculate kernel
	// call this function in neighbourSearch()
	// ker_xx functions always called after neighbourSearch()
	inline void ker_update_numbers();

	real_t m_ker_h;
	real_t m_ker_theta_spline, m_ker_thetah_spline;
	real_t m_ker_theta_spiky, m_ker_thetah_spiky;

    // smoothed kernel and their derivatives
    inline real_t ker_spline(real_t r) const;
    inline real_t ker_spline_grad(real_t r) const;
    inline real_t ker_spiky(real_t r) const;
    inline real_t ker_spiky_grad(real_t r) const;
	inline real_t ker_laplacian_vis(real_t r) const;//see 12

}; // class SphBase


// update internal numbers used to calculate kernel
// call this function in neighbourSearch()
// ker_xx functions always called after neighbourSearch()
inline void SphBase::ker_update_numbers()
{
	if( m_ker_h!=m_TH.smoothRadius_h/2 ){

		m_ker_h = m_TH.smoothRadius_h/2;

		m_ker_theta_spline = (vec_t::dim==3)
			? 1 / real_t(M_PI) / (m_ker_h*m_ker_h*m_ker_h)
			: real_t(10) / 7 / real_t(M_PI) / (m_ker_h*m_ker_h);
		m_ker_thetah_spline = (vec_t::dim==3)
			? 1 / real_t(M_PI) / (m_ker_h*m_ker_h*m_ker_h) / m_ker_h
			: real_t(10) / 7 / real_t(M_PI) / (m_ker_h*m_ker_h) / m_ker_h;

		m_ker_theta_spiky = (vec_t::dim==3)
			? 15 / real_t(M_PI) / (4*m_ker_h*4*m_ker_h*4*m_ker_h)
			: 5 / real_t(M_PI) / (4*m_ker_h*4*m_ker_h);
		m_ker_thetah_spiky = (vec_t::dim==3)
			? 15 / real_t(M_PI) / (4*m_ker_h*4*m_ker_h*4*m_ker_h) / m_ker_h
			: 5 / real_t(M_PI) / (4*m_ker_h*4*m_ker_h) / m_ker_h;
	}

}

/*
 * B-cubic spline kernel W_spline [Mon92] page554
 *
 *       / 1 - 3/2 q^2 + 3/4 q^3		if 0 <= q <= 1
 * theta | 1/4 (2-q)^3				if 1 <= q <= 2
 *       \ 0						otherwise
 * q = r / h
 * theta = 2/3, 10/7pi h^-2, 1/pi h^-3  in 1D, 2D, 3D
*/
inline real_t SphBase::ker_spline(real_t r) const
{
    /*static real_t m_ker_h = m_TH.smoothRadius_h / 2;
    static real_t m_ker_theta_spline = (vec_t::dim==3)
        ? 1 / real_t(M_PI) / (m_ker_h*m_ker_h*m_ker_h)
        : real_t(10) / 7 / real_t(M_PI) / (m_ker_h*m_ker_h);*/

    real_t q = r/m_ker_h, qm2 = 2-q;
    // if( !(q>=0 && q<=2) ) return 0;
    return q<=1
        ? m_ker_theta_spline * ( 1 - real_t(1.5)* q*q + real_t(0.75)* q*q*q )
        : m_ker_theta_spline * ( real_t(0.25)* qm2*qm2*qm2 );

}

/*
 * The graddient of B-cubic spline kernel temp_ker_spline
 * absolute value, the drection towards the center
*/
inline real_t SphBase::ker_spline_grad(real_t r) const
{
    /*static real_t m_ker_h = m_TH.smoothRadius_h / 2;
    static real_t m_ker_thetah_spline = (vec_t::dim==3)
        ? 1 / real_t(M_PI) / (m_ker_h*m_ker_h*m_ker_h) / m_ker_h
        : real_t(10) / 7 / real_t(M_PI) / (m_ker_h*m_ker_h) / m_ker_h;*/

    real_t q = r/m_ker_h, qm2 = 2-q;
    // if( !(q>=0 && q<=2) ) return 0;
    return q<=1
        ? m_ker_thetah_spline * ( 3* q - real_t(2.25)* q*q )
        : m_ker_thetah_spline * real_t(0.75)* qm2*qm2;

}

/*inline real_t SphBase::ker_spline(real_t r) const
{
    static real_t param1 = real_t( 8.0 / ( M_PI*std::pow((double)m_SmoothRadius_h,3.0) ) );
    static real_t param2 = real_t( -48.0 / ( M_PI*std::pow((double)m_SmoothRadius_h,5.0) ) );
    static real_t param3 = real_t( 48.0 / ( M_PI*std::pow((double)m_SmoothRadius_h,6.0) ) );
    static real_t param4 = real_t( 16.0 / ( M_PI*std::pow((double)m_SmoothRadius_h,6.0) ) );
    static real_t h_half = real_t( m_SmoothRadius_h*0.5 );
    // if(r<0 || r>radius_h) return 0;
    return (r<h_half)
        ? ( param1 + param2* r*r + param3* r*r*r )
        : ( param4 * (m_SmoothRadius_h-r)*(m_SmoothRadius_h-r)*(m_SmoothRadius_h-r) );
}
inline real_t SphBase::ker_spline_grad(real_t r) const
{
    static real_t param1 = real_t( 96.0 / ( M_PI*std::pow((double)m_SmoothRadius_h,5.0) ) );
    static real_t param2 = real_t( -144.0 / ( M_PI*std::pow((double)m_SmoothRadius_h,6.0) ) );
    static real_t param3 = real_t( 48.0 / ( M_PI*std::pow((double)m_SmoothRadius_h,6.0) ) );
    static real_t h_half = real_t( m_SmoothRadius_h*0.5 );
    // if(r<0 || r>radius_h) return 0;
    return (r<h_half)
        ? ( param1*r + param2* r*r )
        : ( param3* (m_SmoothRadius_h-r)*(m_SmoothRadius_h-r) );
}*/


/* Spiky kernel W_spiky [Pap96]
 *
 *       / (2-q)^3	if 0 <= q <= 2
 * theta |
 *       \ 0		otherwise
 * q = r / h;
 * theta = 5/pi (4h)^-2, 15/pi (4h)^-3  in 2D, 3D
*/
inline real_t SphBase::ker_spiky(real_t r) const
{
    /*static real_t m_ker_h = m_TH.smoothRadius_h / 2;
    static real_t m_ker_theta_spiky = (vec_t::dim==3)
        ? 15 / real_t(M_PI) / (4*m_ker_h*4*m_ker_h*4*m_ker_h)
        : 5 / real_t(M_PI) / (4*m_ker_h*4*m_ker_h);*/

    real_t qm2 = 2-r/m_ker_h;
    //if( !(q>=0 && q<=2) ) return 0;
    return m_ker_theta_spiky * qm2*qm2*qm2;

}

/* The graddient of spiky kernel W_spiky
 * absolute value, the drection towards the center
*/
inline real_t SphBase::ker_spiky_grad(real_t r) const
{
    /*static real_t m_ker_h = m_TH.smoothRadius_h / 2;
    static real_t m_ker_thetah_spiky = (vec_t::dim==3)
        ? 15 / real_t(M_PI) / (4*m_ker_h*4*m_ker_h*4*m_ker_h) / m_ker_h
        : 5 / real_t(M_PI) / (4*m_ker_h*4*m_ker_h) / m_ker_h;*/

    real_t qm2 = 2-r/m_ker_h;
    //if( !(q>=0 && q<=2) ) return 0;
    return m_ker_thetah_spiky * qm2*qm2;

}

//see 12
inline real_t SphBase::ker_laplacian_vis(real_t r) const {
	//return -3 * r / pow(m_ker_h, 3) + 2 / pow(m_ker_h, 2);	
	return 45 / real_t(M_PI) / pow(2 * m_ker_h, 6) * (2 * m_ker_h - r);
	//return 7.5 / real_t(M_PI) / pow(2 * m_ker_h, 3) * ( (-3*r/ pow(2 * m_ker_h, 3)) + (2/ pow(2 * m_ker_h, 2)) + (2 * m_ker_h/ pow(r, 3)) );
}

/*inline real_t SphBase::ker_spiky(real_t r) const
{
    static real_t param = real_t( 15.0 / ( M_PI*pow((double)m_SmoothRadius_h,6.0) ) );
    // if(r<0 || r>radius_h) return 0;
    return ( param* (m_SmoothRadius_h-r)*(m_SmoothRadius_h-r)*(m_SmoothRadius_h-r) );
}
inline real_t SphBase::ker_spiky_grad(real_t r) const
{
    static real_t param = real_t( 45.0 / ( M_PI*pow((double)m_SmoothRadius_h,6.0) ) );
    //if(r<0 || r>radius_h) return 0;
    return ( param* (m_SmoothRadius_h-r)*(m_SmoothRadius_h-r) );
}*/


#endif // #ifndef SPH_BASE_H_

