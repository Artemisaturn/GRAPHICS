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

#include "WcSph.h"
#include <iomanip>
#include "iostream"
using namespace std;

// override SphBase::runOneStep()
void WcSph::sphStep()
{
#define TIME_RECORD

#ifdef TIME_RECORD
	if( getFrameNumber()==0 ){
		m_clog.setf(std::ios::left);
		//m_clog << "Number\tSearch\t\tWeight\t\tPressure\tForce\t\tconstrainDt\tadaptiveDt\tupFluid\t\tupRigid\t\tDT\t\t\tpercentOfActive\n";
	#ifdef WC_TIMEADAPTIVE
		m_clog << "Number\tTime\t\tselectActiv\tSearch\t\tWeight\t\tPressure\tForce\t\tconstrainDt\tadaptiveDt\tupFluid\t\tupRigid\t\tDT\t\t\tpercentOfActive\n";
	#else
		#ifdef WC_ADT
		m_clog << "Number\tTime\t\tSearch\t\tWeight\t\tPressure\tForce\t\tadaptiveDt\tupFluid\t\tupRigid\t\tDT\n";
		#else
		m_clog << "Number\tTime\t\tSearch\t\tWeight\t\tPressure\tForce\t\tupFluid\t\tupRigid\t\tDT\n";
		#endif
	#endif
	}
	m_clog.width(7); m_clog << m_TH.frameNumber << ' ';
	m_clog.width(11); m_clog << m_TH.systemTime << ' ';

	//#define CALL_TIME(a) this->a()
	double t1, t2;
	#define CALL_TIME(a) t1=omp_get_wtime(); a; t2=omp_get_wtime(); \
						 m_clog.width(11); m_clog <<  (t2-t1)*1000 << ' '
#else
	#define CALL_TIME(a) a
#endif

#ifdef WC_TIMEADAPTIVE
	CALL_TIME(selectActive());
#endif

	//neighbourSearch();
	CALL_TIME(neighbourSearch());

	//updateRigidPartWeight()
	CALL_TIME(updateSolidPartWeight());

	//computePressure();
	CALL_TIME(wc_computePressure());

	//computeForce();
	CALL_TIME(wc_computeForce());

#ifdef WC_TIMEADAPTIVE
	CALL_TIME(wc_constraintDt());
	CALL_TIME(wc_adaptiveDt());
#else
#ifdef WC_ADT
	//CALL_TIME(wc_adaptiveDt());
#endif
#endif

#ifdef WC_TIMEADAPTIVE
	CALL_TIME(wc_updateFluids());
#else
	//updateFluids();
	CALL_TIME(updateFluids());
#endif

	//updateRigids();
	CALL_TIME(updateSolids());

	m_clog.width(11); m_clog << m_TH.dt << ' ';
#ifdef WC_TIMEADAPTIVE
	m_clog.width(11); m_clog << wc_percentOfActive << ' ';
#endif

#ifdef TIME_RECORD
	m_clog << '\n';
#endif

}

// compute fluid particles' pressure, WCSPH or PCISPH
void WcSph::wc_computePressure()
{
	real_t v0 = std::pow( m_TH.spacing_r, vec_t::dim );
	// foreach fluid
	for(int n_f=int(m_Fluids.size()),k=0; k<n_f; ++k){
		std::vector<FluidPart>& f_parts = m_Fluids[k].fluidParticles;
		const std::vector<NeigbStr>& f_neigbs = mg_NeigbOfFluids[k];
		real_t rho0 = m_Fluids[k].restDensity_rho0;
		real_t fm0 = rho0 * v0;
		real_t wm0 = ker_W(0) * fm0;
		real_t B = rho0 * m_TH.soundSpeed_cs*m_TH.soundSpeed_cs / m_DensityPower_gamma;
		int num = int(f_parts.size());
		real_t h = 0.2f;
		
		// foreach particle
	#pragma omp parallel for PARALLEL_SCHEDULE
		for(int i=0; i<num; ++i){
			FluidPart& p_a = f_parts[i];
			vec_t ni = vec_t::O;
		#ifdef WC_TIMEADAPTIVE
			if( p_a.active ){
		#endif
				real_t density = wm0;
				const Neigb* neigbs=f_neigbs[i].neigs; int n=f_neigbs[i].num;
				// forearch neighbour
				for(int j=0; j<n; ++j){
					vec_t grad = vec_t::O;
					if(neigbs[j].pidx.isFluid()){ // fluid neighbour
						const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);
						density += fm0 * ker_W(neigbs[j].dis);
						grad = (p_a.position-p_b.position)*(-ker_W_grad(neigbs[j].dis)/neigbs[j].dis);
						//compute ni(表面张力)
						ni += grad*(fm0/p_b.density);
					}else{ // boundary neighbour
						const BoundPart& p_b = getBoundPartOfIdx(neigbs[j].pidx);
						density += rho0*p_b.volume * ker_W(neigbs[j].dis);
						//ni += grad*(fm0/rho0);
					}
				}
				p_a.density = density;
				ni = ni * m_TH.h;
				p_a.n = ni;
				// Tait's equation, [Mon92], [BT07]
				p_a.presure = B * ( std::pow(density/rho0,m_DensityPower_gamma) - 1 );
				// density error, clamp negative pressure to 0
			#ifdef NEGLECT_NEGATIVE_PRESSURE
				if(p_a.presure<0) p_a.presure=0;
			#endif
		#ifdef WC_TIMEADAPTIVE
			}
		#endif
		}
	}

}

// paritlce-particle interaction, [Mon92], [BT07]
inline void WcSph::wc_fluidPartForce_fsame(
	FluidPart& fa, const FluidPart& fb, const real_t& dis,
	const real_t& fm0, const real_t& alpha, const real_t& gamma )
{
	if( dis==0 ){ ++m_EC.zeroDis; return; }
	/*vec_t xab = fa.position - fb.position;
	vec_t grad = xab * ( -ker_W_grad(dis)/dis );
	// momentum
	fa.acceleration += grad
		* ( -fm0 * (fa.presure/(fa.density*fa.density) + fb.presure/(fb.density*fb.density)) );
	// viscosity
	real_t pro = (fa.velocity-fb.velocity).dot( xab );
	if( pro<0 ){
		real_t nu = 2*alpha*m_TH.smoothRadius_h*m_TH.soundSpeed_cs / ( fa.density + fb.density );
		real_t pi =
			-nu * pro / ( dis*dis + real_t(0.01)*m_TH.smoothRadius_h*m_TH.smoothRadius_h );
		fa.acceleration += grad * ( - fm0 * pi );
	}*/
	vec_t xab = fa.position-fb.position;
	real_t grad = -ker_W_grad(dis)/dis;
	// momentum
	real_t acce = grad
		* ( -fm0 * (fa.presure/(fa.density*fa.density) + fb.presure/(fb.density*fb.density)) );
	// viscosity
	real_t pro = (fa.velocity-fb.velocity).dot( xab );
	if( pro<0 ){
		real_t nu = 2*alpha*m_TH.smoothRadius_h*m_TH.soundSpeed_cs / ( fa.density + fb.density );
		real_t pi =
			-nu * pro / ( dis*dis + real_t(0.01)*m_TH.smoothRadius_h*m_TH.smoothRadius_h );
		acce += grad * ( - fm0 * pi );
	}
	xab *= acce; fa.acceleration += xab;
	// surface tension
	//compute cohesion
	vec_t xab_coh = fa.position-fb.position;
	real_t pai = 3.14f;
	real_t Kij = 2*1000.0f/(fa.density+fb.density);
	real_t C = 0.0f;
	if(2*dis>m_TH.h && dis<=m_TH.h){
		C = (32.0f/(pai*powf(m_TH.h,9.0f)))*powf(m_TH.h-dis,3.0f)*powf(dis,3.0f);
	}else if(dis>0 && 2*dis<=m_TH.h){
		C = (32.0f/(pai*powf(m_TH.h,9.0f)))*(2*powf(m_TH.h-dis,3.0f)*powf(dis,3.0f)-powf(m_TH.h,6.0f)/64.0f);
	}else{
		C = 0.0f;
	}
	real_t acce_coh = -m_TH.r*fm0*C/dis;
	xab_coh *= acce_coh;

	//compute curvature
	real_t acce_cur = -m_TH.r;
	vec_t xab_cur = fa.n-fb.n;
	xab_cur *= acce_cur;

	vec_t acce_st = (xab_coh+xab_cur)*Kij;
	fa.acceleration += acce_st;
	// ...

#ifdef WC_TIMEADAPTIVE
	// d_rho
	fa.drho += fm0 * pro * grad;
#endif

}
// multiphase fluid, [SP08]
inline void WcSph::wc_fluidPartForce_fdiff(
	FluidPart& fa, const FluidPart& fb, const real_t& dis,
	const real_t& fma, const real_t& fmb, const real_t& alpha )
{
	if( dis==0 ){ ++m_EC.zeroDis; return; }
	/*vec_t xab = fa.position - fb.position;
	vec_t grad = xab * ( -ker_W_grad(dis)/dis );
	real_t dalta_a=fa.density/fma, dalta_b=fb.density/fmb;
	// momentum
	fa.acceleration +=
		grad * ( - 1/fma * (fa.presure/(dalta_a*dalta_a) + fb.presure/(dalta_b*dalta_b)) );
	// viscosity
	real_t pro = (fa.velocity-fb.velocity).dot( xab );
	if( pro<0 ){
		real_t nu = 2*alpha*m_TH.smoothRadius_h*m_TH.soundSpeed_cs / ( fa.density + fb.density );
		real_t pi =
			-nu * pro / ( dis*dis + real_t(0.01)*m_TH.smoothRadius_h*m_TH.smoothRadius_h );
		fa.acceleration += grad * ( - (fma+fmb)/2 * pi );
	}*/
	vec_t xab = fa.position-fb.position;
	real_t grad = -ker_W_grad(dis)/dis;
	real_t dalta_a=fa.density/fma, dalta_b=fb.density/fmb;
	// momentum
	real_t acce = grad
		* ( - 1/fma * (fa.presure/(dalta_a*dalta_a) + fb.presure/(dalta_b*dalta_b)) );
	// viscosity
	real_t pro = (fa.velocity-fb.velocity).dot( xab );
	if( pro<0 ){
		real_t nu = 2*alpha*m_TH.smoothRadius_h*m_TH.soundSpeed_cs / ( fa.density + fb.density );
		real_t pi =
			-nu * pro / ( dis*dis + real_t(0.01)*m_TH.smoothRadius_h*m_TH.smoothRadius_h );
		acce += grad * ( - (fma+fmb)/2 * pi );
	}
	xab *= acce; fa.acceleration += xab;

#ifdef WC_TIMEADAPTIVE
	// d_rho
	fa.drho += fma * pro * grad;
#endif

}
// fluid-rigid coupling, [AIS*12]
inline void WcSph::wc_fluidPartForce_bound(
	FluidPart& fa, const BoundPart& rb, const real_t& dis,
	const real_t& frho0, const real_t& r_alpha )
{
	if( dis==0 ){ ++m_EC.zeroDis; return; }
	/*vec_t xab = fa.position - rb.position;
	vec_t grad = xab * ( -ker_W_grad(dis)/dis );
	// momentum
	if(fa.presure>0)
		fa.acceleration +=
			grad * ( - frho0*rb.volume * (fa.presure/(fa.density*fa.density)*2) );
	// viscosity
	real_t pro = (fa.velocity-rb.velocity).dot( xab );
	if( pro<0 ){
		real_t nu = 2*r_alpha*m_TH.smoothRadius_h*m_TH.soundSpeed_cs / ( fa.density*2 );
		real_t pi =
			-nu * pro / ( dis*dis + real_t(0.01)*m_TH.smoothRadius_h*m_TH.smoothRadius_h );
		fa.acceleration += grad * (- frho0*rb.volume * pi );
	}*/
	vec_t xab = fa.position - rb.position;
	real_t grad = -ker_W_grad(dis)/dis, acce=0;
	// momentum
	if(fa.presure>0)
		acce += grad * ( - frho0*rb.volume * (fa.presure/(fa.density*fa.density)*2) );
	// viscosity
	real_t pro = (fa.velocity-rb.velocity).dot( xab );
	if( pro<0 ){
		real_t nu = 2*r_alpha*m_TH.smoothRadius_h*m_TH.soundSpeed_cs / ( fa.density*2 );
		real_t pi =
			-nu * pro / ( dis*dis + real_t(0.01)*m_TH.smoothRadius_h*m_TH.smoothRadius_h );
		acce += grad * (- frho0*rb.volume * pi );
	}
	xab *= acce; fa.acceleration += xab;

#ifdef WC_TIMEADAPTIVE
	// d_rho
	fa.drho += frho0*rb.volume * pro * grad;
#endif
	// surface tension & adhesion
	real_t A = 0.0f;
	vec_t xab_adh = fa.position - rb.position;

	if(2*dis>m_TH.h && dis<=m_TH.h){
		A = 0.007f/(powf(m_TH.h,3.25f))*powf((-4*dis*dis/m_TH.h+6*dis-2*m_TH.h),0.25f);
	}else{
		A = 0.0f;
	}
	real_t acce_adh = -m_TH.bt*frho0*rb.volume*A/dis;
	xab_adh *= acce_adh;
	fa.acceleration += xab_adh;
}
// fluid-rigid coupling, [AIS*12]
inline void WcSph::wc_boundPartForce_f(
	BoundPart& ra, const FluidPart& fb, const real_t& dis,
	const real_t& frho0, const real_t& fm0, const real_t& r_alpha )
{
	if( dis==0 ){ ++m_EC.zeroDis; return; }
	/*vec_t xab = ra.position - fb.position;
	vec_t grad = xab * ( -ker_W_grad(dis)/dis );
	// momentum
	if(fb.presure>0)
		ra.force +=
			grad * ( - fm0*frho0*ra.volume * (fb.presure/(fb.density*fb.density)*2) );
	// viscosity
	real_t pro = (ra.velocity-fb.velocity).dot( xab );
	if( pro<0 ){
		real_t nu = 2*r_alpha*m_TH.smoothRadius_h*m_TH.soundSpeed_cs / ( fb.density*2 );
		real_t pi =
			-nu * pro / ( dis*dis + real_t(0.01)*m_TH.smoothRadius_h*m_TH.smoothRadius_h );
		ra.force += grad * (- fm0*frho0*ra.volume * pi );
	}*/
	vec_t xab = ra.position - fb.position;
	real_t grad = -ker_W_grad(dis)/dis, force=0;
	// momentum
	if(fb.presure>0)
		force += grad * ( - fm0*frho0*ra.volume * (fb.presure/(fb.density*fb.density)*2) );
	// viscosity
	real_t pro = (ra.velocity-fb.velocity).dot( xab );
	if( pro<0 ){
		real_t nu = 2*r_alpha*m_TH.smoothRadius_h*m_TH.soundSpeed_cs / ( fb.density*2 );
		real_t pi =
			-nu * pro / ( dis*dis + real_t(0.01)*m_TH.smoothRadius_h*m_TH.smoothRadius_h );
		force += grad * (- fm0*frho0*ra.volume * pi );
	}
	xab *= force; ra.force += xab;

	real_t bt = 1.0f;
	real_t h = 0.2f;
	real_t A = 0.0f;
	vec_t xab_adh = ra.position - fb.position;

	if(2*dis>m_TH.h && dis<=m_TH.h){
		A = 0.007f/(powf(m_TH.h,3.25f))*powf((-4*dis*dis/m_TH.h+6*dis-2*m_TH.h),0.25f);
	}else{
		A = 0.0f;
	}
	real_t acce_adh = -m_TH.bt*frho0*ra.volume*A/dis;
	xab_adh *= acce_adh;
	ra.force += xab_adh;

}
// compute gravity, pressure and friction force
void WcSph::wc_computeForce()
{
	real_t v0 = std::pow( m_TH.spacing_r, vec_t::dim );
	// foreach fluid
	for(int n_f=int(m_Fluids.size()),k=0; k<n_f; ++k){
		std::vector<FluidPart>& f_parts = m_Fluids[k].fluidParticles;
		const std::vector<NeigbStr>& f_neigbs = mg_NeigbOfFluids[k];
		real_t rho0 = m_Fluids[k].restDensity_rho0;
		real_t fm0 = rho0 * v0;
		real_t alpha = m_Fluids[k].viscosity_alpha;
		real_t gamma = m_Fluids[k].surfaceTension_gamma;
		int num = int(f_parts.size());

		//cout<<m_TH.gravity_g<<endl;
		// foreach particle
	#pragma omp parallel for PARALLEL_SCHEDULE
		for(int i=0; i<num; ++i){
			FluidPart& p_a = f_parts[i];
		#ifdef WC_TIMEADAPTIVE
			if( p_a.active ){
				p_a.t_last = getSystemTime();
				p_a.pos_last = p_a.position;
				p_a.drho = 0;
		#endif
				p_a.acceleration = m_TH.gravity_g;
				
				//p_a.acceleration = vec_t::O;
				const Neigb* neigbs=f_neigbs[i].neigs; int n=f_neigbs[i].num;
				// forearch neighbour
				for(int j=0; j<n; ++j){
					if(neigbs[j].pidx.isFluid()){ // fluid neighbour
						const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);
					#ifdef WC_TIMEADAPTIVE
						((FluidPart&)p_b).dirty = true; 
					#endif
						int idx_b = neigbs[j].pidx.toFluidI();
						// the same fluid
						if(idx_b==k){
							wc_fluidPartForce_fsame(p_a, p_b, neigbs[j].dis, fm0, alpha, gamma);
						// different fluid
						}else{
							real_t fmb = v0 * m_Fluids[idx_b].restDensity_rho0;
							real_t b_alpha = m_Fluids[idx_b].viscosity_alpha;
							wc_fluidPartForce_fdiff(
								p_a, p_b, neigbs[j].dis, fm0, fmb, (alpha+b_alpha)/2);
						}
					}else{ // boundary neighbour
						const BoundPart& p_b = getBoundPartOfIdx(neigbs[j].pidx);
						int idx_b = neigbs[j].pidx.toSolidI();
						real_t r_alpha = m_Solids[idx_b].viscosity_alpha;
						wc_fluidPartForce_bound(p_a, p_b, neigbs[j].dis, rho0, r_alpha);
					}
				}//nneighbour
		#ifdef WC_TIMEADAPTIVE
				// the h in the paper is r, and the smoothing_h is 2h
				real_t v = !(p_a.velocity==vec_t::O)
					? m_Lambda_v*(m_TH.spacing_r/p_a.velocity.length()) : 1;
				real_t f = !(p_a.acceleration==vec_t::O)
					? m_Lambda_f*std::sqrt(m_TH.spacing_r/p_a.acceleration.length()) : 1;
				p_a.dt = p_a.dt_raw = std::min(v, f);
			}
		#else
			#ifdef WC_ADT
				// the h in the paper is r, and the smoothing_h is 2h
				real_t v = !(p_a.velocity==vec_t::O)
					? m_Lambda_v*(m_TH.spacing_r/p_a.velocity.length()) : 1;
				real_t f = !(p_a.acceleration==vec_t::O)
					? m_Lambda_f*std::sqrt(m_TH.spacing_r/p_a.acceleration.length()) : 1;
				p_a.dt = std::min(v, f);
			#endif
		#endif
		}//particle
	}//fluid

	// forearch rigid
	for(int n_r=int(m_Solids.size()),k=0; k<n_r; ++k) if(m_Solids[k].dynamic){
		std::vector<BoundPart>& r_parts = m_Solids[k].boundaryParticles;
		const std::vector<NeigbStr>& r_neigbs = mg_NeigbOfSolids[k];
		real_t alpha = m_Solids[k].viscosity_alpha;
		int num = int(r_parts.size());
		// foreach particle
	#pragma omp parallel for
		for(int i=0; i<num; ++i){
			BoundPart& p_a = r_parts[i];
			p_a.force = vec_t::O;
			const Neigb* neigbs=r_neigbs[i].neigs; int n=r_neigbs[i].num;
			// forearch neighbour
			for(int j=0; j<n; ++j){
				if(neigbs[j].pidx.isFluid()){ // fluid neighbour
					const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);
					int idx_b = neigbs[j].pidx.toFluidI();
					real_t frho0 = m_Fluids[idx_b].restDensity_rho0;
					wc_boundPartForce_f(p_a, p_b, neigbs[j].dis, frho0, frho0*v0, alpha);
				}
			}//neighbour
		}//particle
	}//rigid

}

