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

#ifndef WC_SPH_H_
#define WC_SPH_H_


#include "SphWithIO.h"


class WcSph : public SphWithIO{

public:
#ifdef WC_TIMEADAPTIVE
	//real_t wc_t_v, wc_t_f, wc_t_v_avg, wc_t_f_avg, wc_t_avg;
	real_t wc_percentOfActive;
#endif

	// Constructor
	WcSph(){
		m_DensityPower_gamma = 7;
		m_TH.soundSpeed_cs = 100;
		m_TH.dt = real_t(0.4) * m_TH.spacing_r/m_TH.soundSpeed_cs;
	#ifdef WC_TIMEADAPTIVE
		m_Lambda_v = real_t(0.05);
		m_Lambda_f = real_t(0.025);
		wc_percentOfActive = 1;
	#else
		m_Lambda_v = real_t(0.2); //m_Lambda_v = real_t(0.1);
		m_Lambda_f = real_t(0.1); //m_Lambda_f = real_t(0.05)
	#endif
		//m_DtRatio_k = real_t(1.42); // ~2^0.5
		//m_MaxDt = real_t(0.005); // 5 mS
	}

	// the computation of one step of sph advance, implement in subclasses
	virtual void sphStep();

	// overwrite
	virtual void logInfoOfScene()
	{
		SphBase::logInfoOfScene();

		m_clog << "--------------------- WCSPH ---------\n";
		m_clog << "m_DensityPower_gamma: " << m_DensityPower_gamma << "\n";
	#ifdef WC_TIMEADAPTIVE
		m_clog << "m_Lambda_v: " << m_Lambda_v << "\n";
		m_clog << "m_Lambda_f: " << m_Lambda_f << "\n";
	#endif
		m_clog << "--------------------- WCSPH ---------\n\n";
	}

protected:

	real_t m_DensityPower_gamma;
	real_t m_Lambda_v, m_Lambda_f/*, m_DtRatio_k, m_MaxDt*/; // for adaptive time step

	// compute fluid particles' pressure, WCSPH or PCISPH
	void wc_computePressure();

	// paritlce-particle interaction, [Mon92], [BT07]
	inline void wc_fluidPartForce_fsame(
		FluidPart& fa, const FluidPart& fb, const real_t& dis,
		const real_t& fm0, const real_t& alpha, const real_t& gamma);
	// multiphase fluid, [SP08]
	inline void wc_fluidPartForce_fdiff(
		FluidPart& fa, const FluidPart& fb, const real_t& dis,
		const real_t& fma, const real_t& fmb, const real_t& alpha);
	// fluid-rigid coupling, [AIS*12]
	inline void wc_fluidPartForce_bound(
		FluidPart& fa, const BoundPart& rb, const real_t& dis,
		const real_t& frho0, const real_t& r_alpha);
	// fluid-rigid coupling, [AIS*12]
	inline void wc_boundPartForce_f(
		BoundPart& ra, const FluidPart& fb, const real_t& dis,
		const real_t& frho0, const real_t& fm0, const real_t& r_alpha);
	// compute gravity, pressure and friction force
	void wc_computeForce();

#ifdef WC_TIMEADAPTIVE
	// evaluate that if a particle need to be updated or not
	bool neededUpdate(const FluidPart& p) const{
		return p.t_last+p.dt < getSystemTime() //+m_TH.dt/2
			|| getFrameNumber()%p.sync_factor==0;
		//return true;
	}

	// set particle which satisfies neededUpdate active
	void selectActive(){
		int percentOfActive = 0;
		// foreach fluid
		for(int n_f=int(m_Fluids.size()),k=0; k<n_f; ++k){
			std::vector<FluidPart>& f_parts = m_Fluids[k].fluidParticles;
			int num = int(f_parts.size());
			// foreach particle
		#pragma omp parallel for reduction(+:percentOfActive),PARALLEL_SCHEDULE
			for(int i=0; i<num; ++i){
				FluidPart& p_a = f_parts[i];
				p_a.active = neededUpdate(p_a) ? true : false;
				if(p_a.active) ++percentOfActive;
			}
		}
		wc_percentOfActive = real_t(percentOfActive)/getNumFluidParts();
	}

	// time integration, Euler-Cromer, [IOS*14]
	void wc_updateFluids(){
		real_t t_now = getSystemTime()+m_TH.dt;
		// for each fluid
		for(int n_f=int(m_Fluids.size()),k=0; k<n_f; ++k){
			std::vector<FluidPart>& f_parts = m_Fluids[k].fluidParticles;
			real_t rho0 = m_Fluids[k].restDensity_rho0;
			real_t B = rho0 * m_TH.soundSpeed_cs*m_TH.soundSpeed_cs / m_DensityPower_gamma;
			int num = int(f_parts.size());
			// for each particle
		#pragma omp parallel for PARALLEL_SCHEDULE
			for(int i=0; i<num; ++i){
				FluidPart& p = f_parts[i];
				real_t dt = t_now - p.t_last;
				p.velocity += p.acceleration * m_TH.dt;
				p.position = p.pos_last + p.velocity * dt;
				//*
				p.density += p.drho * m_TH.dt;
				p.presure = B * ( std::pow(p.density/rho0,m_DensityPower_gamma) - 1 );
			#ifdef NEGLECT_NEGATIVE_PRESSURE
				if(p.presure<0) p.presure=0;
			#endif//*/
			}
			//m_Fluids[k].correctOutsideParts(m_TH.spaceMin, m_TH.spaceMax, 0);
			m_Fluids[k].removeOutsideParts(m_TH.spaceMin, m_TH.spaceMax);
		}
	}

	// for i,j that are neighbours, set i.dt/j.dt(assume i.dt>=j.dt) <= m_DtRatio_k
	void wc_constraintDt(){
		// foreach fluid
		for(int n_f=int(m_Fluids.size()),k=0; k<n_f; ++k){
			std::vector<FluidPart>& f_parts = m_Fluids[k].fluidParticles;
			const std::vector<NeigbStr>& f_neigbs = mg_NeigbOfFluids[k];
			int num = int(f_parts.size());
			// foreach particle
		#pragma omp parallel for PARALLEL_SCHEDULE
			for(int i=0; i<num; ++i){
				FluidPart& p_a = f_parts[i];
				if( p_a.active || p_a.dirty ){
					const Neigb* neigbs=f_neigbs[i].neigs; int n=f_neigbs[i].num;
					// forearch neighbour
					for(int j=0; j<n; ++j){
						if(neigbs[j].pidx.isFluid()){ // fluid neighbour
							const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);
							p_a.dt = std::min(p_a.dt, p_b.dt_raw);
						}
					} // neighbour
					p_a.dirty = false;
					p_a.sync_factor = std::max(char(1),char(p_a.dt/m_TH.dt+0.5f));
					//float factor = (float)std::max(1,int(p_a.dt/m_TH.dt));
					//p_a.sync_factor = (char)std::pow(2.0f,int(factor)/std::log(2.0f));
				}
			} // particle
		} // fluid
	}

	// compute adaptive time step, m_TH.dt = min_a(fluid particle a.dt)
	void wc_adaptiveDt(){
		real_t t = 1;
		// foreach fluid
		for(int n_f=int(m_Fluids.size()),k=0; k<n_f; ++k){
			const std::vector<FluidPart>& f_parts = m_Fluids[k].fluidParticles;
			int num = int(f_parts.size());
			// foreach particle
			for(int i=0; i<num; ++i){
				const FluidPart& p_a = f_parts[i];
				t = std::min(t, p_a.dt);
			}
		}
		m_TH.dt = t;
	}

#else
	#ifdef WC_ADT
	// compute adaptive time step, m_TH.dt = min_a(fluid particle a.dt)
	void wc_adaptiveDt(){
		if(m_Fluids.size()==0) return;
		real_t t = 1;
		// foreach fluid
		for(int n_f=int(m_Fluids.size()),k=0; k<n_f; ++k){
			const std::vector<FluidPart>& f_parts = m_Fluids[k].fluidParticles;
			int num = int(f_parts.size());
			// foreach particle
			for(int i=0; i<num; ++i){
				const FluidPart& p_a = f_parts[i];
				t = std::min(t, p_a.dt);
			}
		}
		m_TH.dt = t;
	}
	#endif
#endif

};


#endif WC_SPH_H_ // #ifndef WC_SPH_H_

