#define _SILENCE_STDEXT_HASH_DEPRECATION_WARNINGS 1

#include "DFSPH.h"
#include <iomanip>
#include "iostream"
using namespace std;

// override SphBase::runOneStep()
void DFSPH::sphStep()
{
#define TIME_RECORD

#ifdef TIME_RECORD
	if (getFrameNumber() == 0)
	{
		m_clog.setf(std::ios::left);
		//m_clog << "Number\tSearch\t\tWeight\t\tPressure\tForce\t\tconstrainDt\tadaptiveDt\tupFluid\t\tupRigid\t\tDT\t\t\tpercentOfActive\n";
#ifdef II_TIMEADAPTIVE
		m_clog << "Number\tTime\t\tselectActiv\tSearch\t\tWeight\t\tPressure\tForce\t\tconstrainDt\tadaptiveDt\tupFluid\t\tupRigid\t\tDT\t\t\tpercentOfActive\n";
#else
#ifdef II_ADT
		m_clog << "Number\tTime\t\tSearch\t\tWeight\t\tPressure\tForce\t\tadaptiveDt\tupFluid\t\tupRigid\t\tDT\n";
#else
		m_clog << "Number\tTime\t\tSearch\t\tWeight\t\tPressure\tForce\t\tupFluid\t\tupRigid\t\tDT\n";
#endif
#endif
	}
	m_clog.width(7);
	m_clog << m_TH.frameNumber << ' ';
	m_clog.width(11);
	m_clog << m_TH.systemTime << ' ';

	//#define CALL_TIME(a) this->a()
	double t1, t2;
#define CALL_TIME(a)      \
	t1 = omp_get_wtime(); \
	a;                    \
	t2 = omp_get_wtime(); \
	m_clog.width(11);     \
	m_clog << (t2 - t1) * 1000 << ' '
#else
#define CALL_TIME(a) a
#endif

	/* cycle start */
	if (getFrameNumber() == 0)
	{
		neighbourSearch();
		updateSolidPartWeight();
		DF_constantDensitySolverFirst();
	}
	CALL_TIME(DF_constantDensitySolverSecond());

	CALL_TIME(updateSolids());

	CALL_TIME(neighbourSearch());

	CALL_TIME(updateSolidPartWeight());

	CALL_TIME(DF_constantDensitySolverFirst());

	CALL_TIME(DF_divergenceFreeSolver());
	/* cycle end */

	m_clog.width(11);
	m_clog << m_TH.dt << ' ';
#ifdef II_TIMEADAPTIVE
	m_clog.width(11);
	m_clog << wc_percentOfActive << ' ';
#endif

#ifdef TIME_RECORD
	m_clog << '\n';
#endif
}

/* 
logic of this function: 
1. compute acce_adv
2. compute density and DFalpha
3. iteration of Alogrithm3 in DFSPH
*/

void DFSPH::DF_constantDensitySolverFirst()
{

	/* volume of each particle for the whole system */
	real_t v0 = std::pow(m_TH.spacing_r, vec_t::dim);

	// Dynamic rigid step 1: init p_a.force to zero
	for (int n_r = int(m_Solids.size()), k = 0; k < n_r; ++k)
	{
		std::vector<BoundPart> &r_parts = m_Solids[k].boundaryParticles;
		const std::vector<NeigbStr> &r_neigbs = mg_NeigbOfSolids[k];
		real_t alpha = m_Solids[k].viscosity_alpha;
		int num = int(r_parts.size());
		// foreach particle
#pragma omp parallel for
		for (int i = 0; i < num; ++i)
		{
			BoundPart &p_a = r_parts[i];
			p_a.force = vec_t::O;
		} //particle
	}	  //rigid

	/* compute advection velocity, density, DFalpha*/
	/* loop phase 1 */
	for (int n_f = int(m_Fluids.size()), k = 0; k < n_f; ++k)
	{ // fluid phase "k"

		/* select the particle<vec> and its neighbour<vec><vec> */
		std::vector<FluidPart> &f_parts = m_Fluids[k].fluidParticles;
		const std::vector<NeigbStr> &f_neigbs = mg_NeigbOfFluids[k];

		/* basic attribute for each phase*/
		real_t rho0 = m_Fluids[k].restDensity_rho0;
		real_t fm0 = rho0 * v0;
		real_t wm0 = ker_W(0) * fm0;

		/* particle size */
		int num = int(f_parts.size());

		/* loop particle */
#pragma omp parallel for
		for (int i = 0; i < num; ++i)
		{								 //particle "i" in "k"
			FluidPart &p_a = f_parts[i]; // "i"

			/* initz */
			vec_t grad = vec_t::O; //gradient
			real_t gradScalar = 0; //gradient scalar version
			real_t density = wm0;  // self-contribute density

			/* reset attributes for computation of DFalpha */
			p_a.DFalpha = 0;
			p_a.DFalpha1 = vec_t(0, 0, 0);
			p_a.DFalpha2 = 0;

			/* neighbour<vec> and its size */
			const Neigb *neigbs = f_neigbs[i].neigs;
			int n = f_neigbs[i].num;

			/* loop neighbour */
			for (int j = 0; j < n; ++j)
			{ // neighbour "j" in "i" in "k"

				/* neighbour is fluid */
				if (neigbs[j].pidx.isFluid())
				{
					/* get neighbour */
					const FluidPart &p_b = getFluidPartOfIdx(neigbs[j].pidx);
					/* accumulate neighbour contribution to density */
					density += fm0 * ker_W(neigbs[j].dis);

					/* gradient for Wij */
					gradScalar = -ker_W_grad(neigbs[j].dis);
					grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);
					/* accumulate first part of DFalpha */
					p_a.DFalpha1 += grad * fm0;
					/* accumulate second part of DFalpha */
					p_a.DFalpha2 += pow(gradScalar * fm0, 2);
				}
				/* neighbour is solid */
				else
				{
					/* get neighbour */
					const BoundPart &p_b = getBoundPartOfIdx(neigbs[j].pidx);
					/* accumulate neighbour contribution to density */
					density += rho0 * p_b.volume * ker_W(neigbs[j].dis);

					/* gradient for Wij */
					gradScalar = -ker_W_grad(neigbs[j].dis);
					grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);
					/* accumulate first part of DFalpha */
					p_a.DFalpha1 += grad * rho0 * p_b.volume;
					/* accumulate second part of DFalpha */
					//p_a.DFalpha2 += pow(gradScalar * rho0 * p_b.volume, 2);
				}
			} // end of loop neighbour

			/* obtain density */
			p_a.density = density;

			/* obtain DFalpha */
			p_a.DFalpha = p_a.DFalpha1.len_square() + p_a.DFalpha2; // denominator of DFalpha
			if (p_a.DFalpha < 1.0e-6)
			{ // denominator should be greater than 1e-6
				p_a.DFalpha = 1.0e-6;
			}
			p_a.DFalpha = p_a.density / p_a.DFalpha; // final DFalpha
		}											 // end of loop particle
	}												 // end of first loop phase 1
}

void DFSPH::DF_constantDensitySolverSecond()
{

	/* volume of each particle for the whole system */
	real_t v0 = std::pow(m_TH.spacing_r, vec_t::dim);

	/* compute acce_adv */
	/* gravity is included in this function */
	/* inside is the original ii version */
	DF_forceExceptPressure();

	/* iteration for each phase, first compute advection density,
	then compute DFkappa, finally update (compute) vel_adv
	and compute the new position*/
	/* loop phase 2 */
	for (int n_f = int(m_Fluids.size()), k = 0; k < n_f; ++k)
	{ // fluid phase "k"

		/* select the particle<vec> and its neighbour<vec><vec> */
		std::vector<FluidPart> &f_parts = m_Fluids[k].fluidParticles;
		const std::vector<NeigbStr> &f_neigbs = mg_NeigbOfFluids[k];

		/* basic attribute for each phase*/
		real_t rho0 = m_Fluids[k].restDensity_rho0;
		real_t fm0 = rho0 * v0;

		/* particle size */
		int num = int(f_parts.size());

		/* set the iteration condition */
		real_t errorRateGoal = 0.0001;
		real_t errorRate = 1;
		real_t densitySum = 0;
		int maximumIteration = 50;
		int minimumIteration = 2;
		int currentIteration = 0;

		/* stream correction */
		if (m_TH.enable_vortex == 3)
		{
			if (getFrameNumber() == 0)
			{
				printf("Now running stream function correction\n");
			}
#pragma omp parallel for
			for (int i = 0; i < num; ++i)
			{

				/* initz grad = 0 */
				vec_t grad = vec_t::O;

				/* reference of object $p_a */
				FluidPart &p_a = f_parts[i];

				/* get each neighbour for $p_a */
				const Neigb *neigbs = f_neigbs[i].neigs;
				int n = f_neigbs[i].num;
				/* parameter $tmp is used to accumulate the curl of linear velocity of p_a from its neighbours
				* the vorticity is couputed using the "diffierence curl formulation"
				in [Bender, Jan, et al. "Turbulent Micropolar SPH Fluids with Foam."]  */
				vec_t tmp = vec_t(0, 0, 0);

				for (int j = 0; j < n; ++j)
				{
					/* if the neighbour is fluid */
					if (neigbs[j].pidx.isFluid())
					{
						const FluidPart &p_b = getFluidPartOfIdx(neigbs[j].pidx);
						grad = (p_a.position - p_b.position) * (-ker_W_grad(neigbs[j].dis) / neigbs[j].dis);
						tmp += ((p_a.velocity - p_b.velocity) * (1 / p_a.density * fm0)).cross(grad);
					}
					/* else the neighbour is boundary */
					else
					{
						/*const BoundPart& p_b = getBoundPartOfIdx(neigbs[j].pidx);
						grad = (p_a.position - p_b.position)*(-ker_W_grad(neigbs[j].dis) / neigbs[j].dis);
						tmp += ((p_a.velocity - p_b.singleVelocity)*(1 / p_a.density * (rho0 * p_b.volume))).cross(grad);*/
					}
				}
				p_a.n0Vorticity = tmp;
			}
		}

		/* vorticity refinement TRRV */
		if (m_TH.enable_vortex == 1)
		{

			if (getFrameNumber() == 0)
			{
				printf("Now running TRRV\n");
			}

			/* 1st loop of TRRV: compute vorticity for each fluid partilce
			* the particel is $p_a
			* curl of linear velocity for $p_a is $p_a.vortex2
			* the neighbour particles of $p_a are $p_b
			* foreach fluid particle in one fluid phase
			*/

#pragma omp parallel for
			for (int i = 0; i < num; ++i)
			{

				/* initz grad = 0 */
				vec_t grad = vec_t::O;

				/* reference of object $p_a */
				FluidPart &p_a = f_parts[i];
				/* initz parameters for particle $p_a */
				p_a.dOmega = vec_t(0, 0, 0);

				/* get each neighbour for $p_a */
				const Neigb *neigbs = f_neigbs[i].neigs;
				int n = f_neigbs[i].num;
				/* parameter $tmp is used to accumulate the curl of linear velocity of p_a from its neighbours
				* the vorticity is couputed using the "diffierence curl formulation"
				in [Bender, Jan, et al. "Turbulent Micropolar SPH Fluids with Foam."]  */
				vec_t tmp = vec_t(0, 0, 0);

				for (int j = 0; j < n; ++j)
				{
					/* if the neighbour is fluid */
					if (neigbs[j].pidx.isFluid())
					{
						const FluidPart &p_b = getFluidPartOfIdx(neigbs[j].pidx);
						grad = (p_a.position - p_b.position) * (-ker_W_grad(neigbs[j].dis) / neigbs[j].dis);
						tmp += ((p_a.velocity - p_b.velocity) * (1 / p_a.density * fm0)).cross(grad);
					}
					/* else the neighbour is boundary */
					else
					{
						const BoundPart &p_b = getBoundPartOfIdx(neigbs[j].pidx);
						grad = (p_a.position - p_b.position) * (-ker_W_grad(neigbs[j].dis) / neigbs[j].dis);
						tmp += ((p_a.velocity - p_b.singleVelocity) * (1 / p_a.density * fm0)).cross(grad);
					}
				}
				/* get the final vorticity
				* should notice the vortex is averaged to avoid instability issue */
				//p_a.omega = (p_a.omega + tmp)*0.5;

				p_a.omega = tmp;
			}

			/* 2nd loop of TRRV: calculate the diffierence of vorticity between this and the previous time step
			* the vorticity diffierence is $p_a.vorticity_diff
			* the vorticity from previous time step is $p_a.vorticity_m
			* $p_a.vorticity_ratio was used to modify the roughness of the turbulence, now is useless
			* $m_TH.alpha is the new adjustment parameter
			* the angular velocity used to refine linear velocity is $p_a.dOmega
			*/
#pragma omp parallel for
			for (int i = 0; i < num; ++i)
			{
				/* initz gradient */
				vec_t grad = vec_t::O;
				/* reference of object $p_a */
				FluidPart &p_a = f_parts[i];

				/* get each neighbour for $p_a */
				const Neigb *neigbs = f_neigbs[i].neigs;
				int n = f_neigbs[i].num;
				/* $tmp here is used to accumulate the dOmega from neighbours
				* this is according to the vorticity of N-S equations (VNS) */
				vec_t tmp = vec_t(0, 0, 0);
				vec_t tmp1 = vec_t(0, 0, 0);
				vec_t tmp2 = vec_t(0, 0, 0);
				for (int j = 0; j < n; ++j)
				{
					/* if the neighbour is fluid */
					if (neigbs[j].pidx.isFluid())
					{
						const FluidPart &p_b = getFluidPartOfIdx(neigbs[j].pidx);
						vec_t xab = p_a.position - p_b.position;
						grad = (xab) * (-ker_W_grad(neigbs[j].dis) / neigbs[j].dis);
						/* diffusion term of VNS */
						real_t pro = (p_a.omega - p_b.omega).dot(xab);
						tmp1 += grad * pro / (neigbs[j].dis * neigbs[j].dis + real_t(0.01) * m_TH.smoothRadius_h * m_TH.smoothRadius_h) * fm0 / p_b.density * 10 * m_TH.nu;
						/* stretching term of VNS */
						for (int jj = 0; jj < 3; jj++)
						{
							/*vec_t velocity_gradient_one_d = grad * (fm0 * (p_a.velocity[jj] / (p_a.density*p_a.density) + p_b.velocity[jj] / (p_b.density*p_b.density)));
							tmp2[jj] += p_a.omega.dot(velocity_gradient_one_d);*/
							/*real_t omega_follow_velocity_one_d = p_a.omega.dot(grad * (fm0 * (p_a.velocity[jj] / (p_a.density*p_a.density) + p_b.velocity[jj] / (p_b.density*p_b.density))));
							real_t omega_follow_omega_one_d = p_a.omega.dot(grad * (fm0 * (p_a.omega[jj] / (p_a.density*p_a.density) + p_b.omega[jj] / (p_b.density*p_b.density))));*/
							real_t omega_follow_velocity_one_d = p_a.omega.dot(grad * (fm0 * (p_a.velocity[jj] / (p_a.density * p_a.density) + p_b.velocity[jj] / (p_b.density * p_b.density))));
							tmp2[jj] += omega_follow_velocity_one_d;
						}
					}
				}
				tmp = tmp1 + tmp2;
				p_a.dOmega = -tmp * m_TH.dt;
			}

			/* 3rd loop of TRRV: refine linear velocity using dOmega
			* linear velocity is refined through Rankine vortex model
			*/
#pragma omp parallel for
			for (int i = 0; i < num; ++i)
			{
				/* initz gradient */
				vec_t grad = vec_t::O;
				/* reference of object $p_a */
				FluidPart &p_a = f_parts[i];
				/* get each neighbour for $p_a */
				const Neigb *neigbs = f_neigbs[i].neigs;
				int n = f_neigbs[i].num;
				// for each nieghbour
				for (int j = 0; j < n; ++j)
				{
					if (neigbs[j].pidx.isFluid())
					{ // fluid neighbour
						const FluidPart &p_b = getFluidPartOfIdx(neigbs[j].pidx);
						/* $pp is the distance from a to b */
						vec_t pp = p_a.position - p_b.position;
						/* $tmp here represents converted linear velocity from dOmega
						and is the final value of TRRV */
						vec_t tmp = (p_b.dOmega / 2).cross(pp) * pow(m_TH.spacing_r / 2 / pp.length(), 2);
						/* the refined value is then added to vel_adv */
						p_a.velocity += tmp * m_TH.alpha;
					}
				}
			}
		}
		/* end of vorticity refinement TRRV */

		/* vorticity refinement micropolar */
		if (m_TH.enable_vortex == 2)
		{

			if (getFrameNumber() == 0)
			{
				printf("Now running micropolar\n");
			}

			//1.1th loop forearch fluid particle reset parameters for micropolar approach
#pragma omp parallel for
			for (int i = 0; i < num; ++i)
			{
				FluidPart &p_a = f_parts[i];
				p_a.vorticity = vec_t(0, 0, 0);
				p_a.laplacian_angular_velocity = vec_t(0, 0, 0);
				p_a.vortical_angular_velocity = vec_t(0, 0, 0);
			}
			//1.15th loop forearch rigid particle update new velocity
//#pragma omp parallel for
//			for (int i = 0; i < num; ++i) {
//				FluidPart& p_a = f_parts[i];
//				p_a.vorticity = vec_t(0, 0, 0);
//				p_a.laplacian_angular_velocity = vec_t(0, 0, 0);
//				p_a.vortical_angular_velocity = vec_t(0, 0, 0);
//			}
//---------------------------------------------------------------------------------------------------------------------------------------
//1.4th loop forearch fluid particle compute vorticity
#pragma omp parallel for
			for (int i = 0; i < num; ++i)
			{
				vec_t grad = vec_t::O;
				FluidPart &p_a = f_parts[i];
				// forearch neighbour, compute vorticity
				const Neigb *neigbs = f_neigbs[i].neigs;
				int n = f_neigbs[i].num;
				for (int j = 0; j < n; ++j)
				{
					if (neigbs[j].pidx.isFluid())
					{ // fluid neighbour
						const FluidPart &p_b = getFluidPartOfIdx(neigbs[j].pidx);
						grad = (p_a.position - p_b.position) * (-ker_W_grad(neigbs[j].dis) / neigbs[j].dis);
						p_a.vorticity += ((p_a.velocity - p_b.velocity) * (1 / p_a.density * fm0)).cross(grad);
					}
					else
					{ // boundary neighbour
						const BoundPart &p_b = getBoundPartOfIdx(neigbs[j].pidx);
						grad = (p_a.position - p_b.position) * (-ker_W_grad(neigbs[j].dis) / neigbs[j].dis);
						p_a.vorticity += ((p_a.velocity - p_b.velocity) * (1 / p_a.density * (p_b.volume * rho0))).cross(grad);
						//psi
					}
				}
			}

			//1.5th loop for Laplacian angular_velocity
#pragma omp parallel for
			for (int i = 0; i < num; ++i)
			{
				vec_t grad = vec_t::O;
				FluidPart &p_a = f_parts[i];

				// forearch neighbour, compute Laplacian angular_velocity
				const Neigb *neigbs = f_neigbs[i].neigs;
				int n = f_neigbs[i].num;
				for (int j = 0; j < n; ++j)
				{
					if (neigbs[j].pidx.isFluid())
					{ // fluid neighbour
						const FluidPart &p_b = getFluidPartOfIdx(neigbs[j].pidx);
						vec_t xab = p_a.position - p_b.position;
						real_t pro = (p_a.angular_velocity - p_b.angular_velocity).dot(xab);
						grad = (p_a.position - p_b.position) * (-ker_W_grad(neigbs[j].dis) / neigbs[j].dis);
						p_a.laplacian_angular_velocity +=
							grad * pro / (neigbs[j].dis * neigbs[j].dis + real_t(0.01) * m_TH.smoothRadius_h * m_TH.smoothRadius_h) * fm0 / p_b.density * 10;
					}
					else
					{ // boundary neighbour
						const BoundPart &p_b = getBoundPartOfIdx(neigbs[j].pidx);
						grad = (p_a.position - p_b.position) * (-ker_W_grad(neigbs[j].dis) / neigbs[j].dis);
						vec_t xab = p_a.position - p_b.position;
						real_t pro = (p_a.angular_velocity - p_b.angular_velocity).dot(xab);
						p_a.laplacian_angular_velocity +=
							grad * pro / (neigbs[j].dis * neigbs[j].dis + real_t(0.01) * m_TH.smoothRadius_h * m_TH.smoothRadius_h) * p_b.volume * 10;
						//psi
					}
				}
			}

			//1.6th loop forearch particle update angular velocity
#pragma omp parallel for
			for (int i = 0; i < num; ++i)
			{
				FluidPart &p_a = f_parts[i];
				p_a.angular_velocity += ((p_a.laplacian_angular_velocity * m_TH.zeta) +
										 (p_a.vorticity - (p_a.angular_velocity * 2)) * (m_TH.nu + m_TH.nu_t)) *
										0.5 * m_TH.dt;
			}

			//---------------------------------------------------------------------------------------------------------------------------------------
			//1.2th loop forearch particle compute vortical_angular_velocity
#pragma omp parallel for
			for (int i = 0; i < num; ++i)
			{
				vec_t grad = vec_t::O;
				FluidPart &p_a = f_parts[i];
				// forearch neighbour, compute vortical_angular_velocity
				const Neigb *neigbs = f_neigbs[i].neigs;
				int n = f_neigbs[i].num;
				for (int j = 0; j < n; ++j)
				{
					if (neigbs[j].pidx.isFluid())
					{ // fluid neighbour
						const FluidPart &p_b = getFluidPartOfIdx(neigbs[j].pidx);
						grad = (p_a.position - p_b.position) * (-ker_W_grad(neigbs[j].dis) / neigbs[j].dis);
						p_a.vortical_angular_velocity += ((p_a.angular_velocity - p_b.angular_velocity) * (1 / p_a.density * fm0)).cross(grad);
					}
					else
					{ // boundary neighbour
						const BoundPart &p_b = getBoundPartOfIdx(neigbs[j].pidx);
						grad = (p_a.position - p_b.position) * (-ker_W_grad(neigbs[j].dis) / neigbs[j].dis);
						p_a.vortical_angular_velocity += ((p_a.angular_velocity - p_b.angular_velocity) * (1 / p_a.density * (p_b.volume * rho0))).cross(grad);
						//psi
					}
				}
			}
			//1.3th loop forearch particle update vel_adv
#pragma omp parallel for
			for (int i = 0; i < num; ++i)
			{
				FluidPart &p_a = f_parts[i];
				p_a.velocity += p_a.vortical_angular_velocity * (m_TH.nu + m_TH.nu_t) * m_TH.dt;
			}
		}
		/* end of vorticity refinement micropolar */

		/* loop particle 2.0 */
		/* compute vel_adv */
#pragma omp parallel for
		for (int i = 0; i < num; ++i)
		{								 //particle "i" in "k"
			FluidPart &p_a = f_parts[i]; // "i"
			/* obtain advection velocity */
			p_a.vel_adv = p_a.velocity + p_a.acce_adv * m_TH.dt;
		} // end of loop particle 2.0

		/* start the iteration */
		while (errorRate > errorRateGoal || currentIteration < minimumIteration)
		{

			/* reset for each iteration */
			densitySum = 0;

			/* loop particle 2.1 */
			/* update (compute) density_adv */
#pragma omp parallel for
			for (int i = 0; i < num; ++i)
			{								 //particle "i" in "k"
				FluidPart &p_a = f_parts[i]; // "i"

				/* initz */
				vec_t grad = vec_t::O; //gradient
				real_t gradScalar = 0; //gradient scalar version
				p_a.density_adv = p_a.density;

				/* neighbour<vec> and its size */
				const Neigb *neigbs = f_neigbs[i].neigs;
				int n = f_neigbs[i].num;

				/* loop neighbour */
				for (int j = 0; j < n; ++j)
				{ // neighbour "j" in "i" in "k"
					/* neighbour is fluid */
					if (neigbs[j].pidx.isFluid())
					{
						/* get neighbour */
						const FluidPart &p_b = getFluidPartOfIdx(neigbs[j].pidx);

						/* gradient for Wij */
						gradScalar = -ker_W_grad(neigbs[j].dis);
						grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

						/* accumulate (obtain) change of density_adv */
						p_a.density_adv += grad.dot(p_a.vel_adv - p_b.vel_adv) * fm0 * m_TH.dt;
					}
					/* neighbour is solid */
					else
					{
						/* get neighbour */
						const BoundPart &p_b = getBoundPartOfIdx(neigbs[j].pidx);

						/* gradient for Wij */
						gradScalar = -ker_W_grad(neigbs[j].dis);
						grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

						/* accumulate (obtain) change of density_adv */
						p_a.density_adv += grad.dot(p_a.vel_adv - p_b.singleVelocity) * rho0 * p_b.volume * m_TH.dt; //potential error because of p_b.vel_adv (solid)
					}
				} // end of loop neighbour

				/* adjust the density_adv */
				if (p_a.density_adv < rho0)
				{ // when lack of neighbours, p_a should be regarded as restDensity
					p_a.density_adv = rho0;
				}
			} // end of loop particle 2.1

			/* densitySum cannot be parelled */
			for (int i = 0; i < num; ++i)
			{								 //particle "i" in "k"
				FluidPart &p_a = f_parts[i]; // "i"
				densitySum += p_a.density_adv;
			}

			/* loop particle 2.2 */
			/* compute DFkappa and sumDensity*/
			//#pragma omp parallel for
			for (int i = 0; i < num; ++i)
			{								 //particle "i" in "k"
				FluidPart &p_a = f_parts[i]; // "i"

				/* obtain DFkappa */
				p_a.DFkappa = (p_a.density_adv - rho0) / pow(m_TH.dt, 2) * p_a.DFalpha;

			} // end of loop particle 2.2

			/* loop particle 2.3 */
			/* update (compute) vel_adv */
#pragma omp parallel for
			for (int i = 0; i < num; ++i)
			{								 //particle "i" in "k"
				FluidPart &p_a = f_parts[i]; // "i"

				/* initz */
				vec_t grad = vec_t::O; //gradient
				real_t gradScalar = 0; //gradient scalar version

				/* neighbour<vec> and its size */
				const Neigb *neigbs = f_neigbs[i].neigs;
				int n = f_neigbs[i].num;

				/* loop neighbour */
				for (int j = 0; j < n; ++j)
				{ // neighbour "j" in "i" in "k"
					/* neighbour is fluid */
					if (neigbs[j].pidx.isFluid())
					{
						/* get neighbour */
						const FluidPart &p_b = getFluidPartOfIdx(neigbs[j].pidx);

						/* gradient for Wij */
						gradScalar = -ker_W_grad(neigbs[j].dis);
						grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

						/* accumulate (obtain) change of vel_adv */
						p_a.vel_adv = p_a.vel_adv - (grad * fm0 * ((p_a.DFkappa / p_a.density) + (p_b.DFkappa / p_b.density)) * m_TH.dt);
					}
					/* neighbour is solid */
					// Dynamic Rigid step 4: 将固体粒子处理做如下修改
					else
					{
						/* get neighbour */
						BoundPart &p_b = m_Solids[neigbs[j].pidx.toSolidI()].boundaryParticles[neigbs[j].pidx.i];

						/* gradient for Wij */
						gradScalar = -ker_W_grad(neigbs[j].dis);
						grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

						/* accumulate (obtain) change of vel_adv */
						vec_t acc_p = grad * rho0 * p_b.volume * ((p_a.DFkappa / p_a.density));
						p_a.vel_adv = p_a.vel_adv - (acc_p * m_TH.dt);
						p_b.force += acc_p;
					}
				} // end of loop neighbour
			}	  // end of loop particle 2.3

			/* recompute density error rate */
			errorRate = ((densitySum / num) - rho0) / rho0;
			/* update current iteration */
			currentIteration++;

		} //end of iteration
		  //printf("cur iter: %d \n", currentIteration);

		/* loop particle 2.4 */
		/* update (compute) velocity and update (compute) position */
#pragma omp parallel for
		for (int i = 0; i < num; ++i)
		{								 //particle "i" in "k"
			FluidPart &p_a = f_parts[i]; // "i"

			/* obtain velocity */
			p_a.velocity = p_a.vel_adv;

			/* obtain position */
			p_a.position += p_a.velocity * m_TH.dt;
		} // end of loop particle 2.4

		/* Remove Outside Particles */
		m_Fluids[k].removeOutsideParts(m_TH.spaceMin, m_TH.spaceMax);

	} // end of second loop phase 2

	// Dynamic Rigid step 5: 将 加速度/密度[m^4/s^2*kg] 变为 力[N]
	for (int n_r = int(m_Solids.size()), k = 0; k < n_r; ++k)
	{
		std::vector<BoundPart> &r_parts = m_Solids[k].boundaryParticles;
		const std::vector<NeigbStr> &r_neigbs = mg_NeigbOfSolids[k];
		real_t alpha = m_Solids[k].viscosity_alpha;
		int num = int(r_parts.size());
		// foreach particle
		real_t fm0 = (v0 * m_Fluids[0].restDensity_rho0);
#pragma omp parallel for
		for (int i = 0; i < num; ++i)
		{
			BoundPart &p_a = r_parts[i];
			p_a.force = p_a.force * fm0;
		} //particle
	}	  //rigid
}

void DFSPH::DF_divergenceFreeSolver()
{

	/* volume of each particle for the whole system */
	real_t v0 = std::pow(m_TH.spacing_r, vec_t::dim);

	/* loop phase */
	for (int n_f = int(m_Fluids.size()), k = 0; k < n_f; ++k)
	{ // fluid phase "k"

		/* select the particle<vec> and its neighbour<vec><vec> */
		std::vector<FluidPart> &f_parts = m_Fluids[k].fluidParticles;
		const std::vector<NeigbStr> &f_neigbs = mg_NeigbOfFluids[k];

		/* basic attribute for each phase*/
		real_t rho0 = m_Fluids[k].restDensity_rho0;
		real_t fm0 = rho0 * v0;

		/* particle size */
		int num = int(f_parts.size());

		/* set the iteration condition */
		real_t errorRateGoal = 2.5;
		real_t divergenceDeviationAver = 0;
		int maximumIteration = 30;
		int minimumIteration = 1;
		int currentIteration = 0;

		/* start the iteration */
		while (divergenceDeviationAver > errorRateGoal || currentIteration < minimumIteration)
		{

			if (currentIteration > maximumIteration)
				break;

			/* reset for each iteration */
			divergenceDeviationAver = 0;

			/* loop particle 1 */
			/* compute divergence deviation and compute DFkappaV for each particle */
#pragma omp parallel for
			for (int i = 0; i < num; ++i)
			{								 //particle "i" in "k"
				FluidPart &p_a = f_parts[i]; // "i"

				/* initz */
				vec_t grad = vec_t::O;		 //gradient
				real_t gradScalar = 0;		 //gradient scalar version
				p_a.divergenceDeviation = 0; // divergence deviation of the particle

				/* neighbour<vec> and its size */
				const Neigb *neigbs = f_neigbs[i].neigs;
				int n = f_neigbs[i].num;

				/* loop neighbour */
				for (int j = 0; j < n; ++j)
				{ // neighbour "j" in "i" in "k"
					/* neighbour is fluid */
					if (neigbs[j].pidx.isFluid())
					{
						/* get neighbour */
						const FluidPart &p_b = getFluidPartOfIdx(neigbs[j].pidx);

						/* gradient for Wij */
						gradScalar = -ker_W_grad(neigbs[j].dis);
						grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

						/* accumulate (obtain) divergenceDeviation */
						p_a.divergenceDeviation += grad.dot(p_a.vel_adv - p_b.vel_adv) * fm0;
						//p_a.divergenceDeviation += -grad.dot(p_a.vel_adv - p_b.vel_adv)*p_a.density;
					}
					/* neighbour is solid */
					else
					{
						/* get neighbour */
						const BoundPart &p_b = getBoundPartOfIdx(neigbs[j].pidx);

						/* gradient for Wij */
						gradScalar = -ker_W_grad(neigbs[j].dis);
						grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

						/* accumulate (obtain) divergenceDeviation */
						p_a.divergenceDeviation += grad.dot(p_a.vel_adv - p_b.singleVelocity) * (rho0 * p_b.volume); //potential error because of p_b.vel_adv (solid)
																													 //p_a.divergenceDeviation += -grad.dot(p_a.vel_adv - p_b.singleVelocity)*p_a.density;
					}
				} // end of loop neighbour

				/* divergenceDeviation must greater than or equal to zero */
				/* no deviation when less than rest desity */
				if (p_a.divergenceDeviation < 0 || (p_a.density + p_a.divergenceDeviation * m_TH.dt) < rho0)
				{
					p_a.divergenceDeviation = 0;
				}

				/* obtain DFkappaV */
				p_a.DFkappaV = p_a.divergenceDeviation * p_a.DFalpha / m_TH.dt;
			} // end of loop particle 1

			/* loop particle 2 */
			/* compute vel_adv for each particle */
#pragma omp parallel for
			for (int i = 0; i < num; ++i)
			{								 //particle "i" in "k"
				FluidPart &p_a = f_parts[i]; // "i"

				/* initz */
				vec_t grad = vec_t::O; //gradient
				real_t gradScalar = 0; //gradient scalar version

				/* neighbour<vec> and its size */
				const Neigb *neigbs = f_neigbs[i].neigs;
				int n = f_neigbs[i].num;

				/* loop neighbour */
				for (int j = 0; j < n; ++j)
				{ // neighbour "j" in "i" in "k"
					/* neighbour is fluid */
					if (neigbs[j].pidx.isFluid())
					{
						/* get neighbour */
						const FluidPart &p_b = getFluidPartOfIdx(neigbs[j].pidx);

						/* gradient for Wij */
						gradScalar = -ker_W_grad(neigbs[j].dis);
						grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

						/* accumulate (obtain) change of vel_adv */
						p_a.vel_adv = p_a.vel_adv - (grad * fm0 * ((p_a.DFkappaV / p_a.density) + (p_b.DFkappaV / p_b.density)) * m_TH.dt);
					}
					/* neighbour is solid */
					else
					{
						/* get neighbour */
						const BoundPart &p_b = getBoundPartOfIdx(neigbs[j].pidx);

						/* gradient for Wij */
						gradScalar = -ker_W_grad(neigbs[j].dis);
						grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

						/* accumulate (obtain) change of vel_adv */
						p_a.vel_adv = p_a.vel_adv - (grad * rho0 * p_b.volume * ((p_a.DFkappaV / p_a.density)) * m_TH.dt);
					}
				} // end of loop neighbour
			}	  // end of loop particle 2

			/* loop particle 3 */
			/* compute average divergence deviation */
			/* cannot be paralleled */
			for (int i = 0; i < num; ++i)
			{								 //particle "i" in "k"
				FluidPart &p_a = f_parts[i]; // "i"

				/* obtain divergenceDeviationAver sum */
				divergenceDeviationAver += p_a.divergenceDeviation;
			} // end of loop particle 3
			/* obtain divergenceDeviationAver */
			divergenceDeviationAver = divergenceDeviationAver / num;

			currentIteration++;

		} // end of the iteration

		/* loop particle 4 */
		/* update (compute) velocity */
#pragma omp parallel for
		for (int i = 0; i < num; ++i)
		{								 //particle "i" in "k"
			FluidPart &p_a = f_parts[i]; // "i"
			/* obtain velocity */
			p_a.velocity = p_a.vel_adv;
		} // end of loop particle 4

		/* stream correction */
		if (m_TH.enable_vortex == 3)
		{
#pragma omp parallel for
			for (int i = 0; i < num; ++i)
			{

				/* initz grad = 0 */
				vec_t grad = vec_t::O;
				/* reference of object $p_a */
				FluidPart &p_a = f_parts[i];
				/* get each neighbour for $p_a */
				const Neigb *neigbs = f_neigbs[i].neigs;
				int n = f_neigbs[i].num;
				vec_t tmp = vec_t(0, 0, 0);
				vec_t tmp1 = vec_t(0, 0, 0);
				vec_t tmp2 = vec_t(0, 0, 0);
				bool flag = false;
				for (int j = 0; j < n; ++j)
				{
					/* if the neighbour is fluid */
					if (neigbs[j].pidx.isFluid())
					{
						const FluidPart &p_b = getFluidPartOfIdx(neigbs[j].pidx);
						vec_t xab = p_a.position - p_b.position;
						grad = (xab) * (-ker_W_grad(neigbs[j].dis) / neigbs[j].dis);
						tmp += ((p_a.velocity - p_b.velocity) * (1 / p_a.density * fm0)).cross(grad);
						real_t pro = (p_a.omega - p_b.omega).dot(xab);
						tmp1 += grad * pro / (neigbs[j].dis * neigbs[j].dis + real_t(0.01) * m_TH.smoothRadius_h * m_TH.smoothRadius_h) * fm0 / p_b.density * 10 * m_TH.nu;
						for (int jj = 0; jj < 3; jj++)
						{
							real_t omega_follow_velocity_one_d = p_a.omega.dot(grad * (fm0 * (p_a.velocity[jj] / (p_a.density * p_a.density) + p_b.velocity[jj] / (p_b.density * p_b.density))));
							tmp2[jj] += omega_follow_velocity_one_d;
						}
					}
					/* else the neighbour is boundary */
					else
					{
						flag = true;
						break;
						/*const BoundPart& p_b = getBoundPartOfIdx(neigbs[j].pidx);
						vec_t xab = p_a.position - p_b.position;
						grad = (xab)*(-ker_W_grad(neigbs[j].dis) / neigbs[j].dis);
						tmp += ((p_a.velocity - p_b.singleVelocity)*(1 / p_a.density * (rho0 * p_b.volume))).cross(grad);
						real_t pro = (p_a.omega - vec_t(0,0,0)).dot(xab);
						tmp1 += grad * pro / (neigbs[j].dis*neigbs[j].dis + real_t(0.01)*m_TH.smoothRadius_h*m_TH.smoothRadius_h) * (rho0*p_b.volume) / p_a.density * 10 * m_TH.nu;
						for (int jj = 0; jj < 3; jj++) {
							real_t omega_follow_velocity_one_d = p_a.omega.dot(grad * (fm0 * (p_a.velocity[jj] / (p_a.density*p_a.density) + p_b.singleVelocity[jj] / (rho0*rho0))));
							tmp2[jj] += omega_follow_velocity_one_d;
						}*/
					}
				}
				if (flag)
				{
					p_a.deltaVorticity = vec_t(0, 0, 0);
					continue;
				}
				p_a.n0Vorticity += (tmp1 + tmp2) * m_TH.dt;
				p_a.n1Vorticity = tmp;
				p_a.deltaVorticity = p_a.n0Vorticity - p_a.n1Vorticity;
				if (p_a.n0Vorticity.len_square() < p_a.n1Vorticity.len_square())
				{
					p_a.deltaVorticity = vec_t(0, 0, 0);
				}
			}
#pragma omp parallel for
			for (int i = 0; i < num; ++i)
			{
				/* initz grad = 0 */
				vec_t grad = vec_t::O;
				/* reference of object $p_a */
				FluidPart &p_a = f_parts[i];
				p_a.psi = vec_t(0, 0, 0);
				/* get each neighbour for $p_a */
				const Neigb *neigbs = f_neigbs[i].neigs;
				int n = f_neigbs[i].num;

				for (int j = 0; j < n; ++j)
				{
					const FluidPart &p_b = getFluidPartOfIdx(neigbs[j].pidx);
					if (neigbs[j].pidx.isFluid())
					{
						p_a.psi += p_b.deltaVorticity * v0 / 4 / 3.1415926 / neigbs[j].dis;
					}
				}
			}
#pragma omp parallel for
			for (int i = 0; i < num; ++i)
			{
				vec_t tmp = vec_t(0, 0, 0);
				/* initz grad = 0 */
				vec_t grad = vec_t::O;
				/* reference of object $p_a */
				FluidPart &p_a = f_parts[i];

				const Neigb *neigbs = f_neigbs[i].neigs;
				int n = f_neigbs[i].num;

				for (int j = 0; j < n; ++j)
				{
					const FluidPart &p_b = getFluidPartOfIdx(neigbs[j].pidx);
					if (neigbs[j].pidx.isFluid())
					{
						vec_t xab = p_a.position - p_b.position;
						grad = (xab) * (-ker_W_grad(neigbs[j].dis) / neigbs[j].dis);
						tmp += ((p_a.psi - p_b.psi) * (1 / p_a.density * fm0)).cross(grad);
					}
				}
				p_a.velocity += tmp;
			}
		}
	} // end of loop phase
}

// compute fluid particles' pressure, WCSPH or PCISPH
void DFSPH::ii_computePressure()
{
	real_t v0 = std::pow(m_TH.spacing_r, vec_t::dim);

	//compute acce_adv;
	ii_forceExceptPressure();

	// foreach fluid
	for (int n_f = int(m_Fluids.size()), k = 0; k < n_f; ++k)
	{
		std::vector<FluidPart> &f_parts = m_Fluids[k].fluidParticles;
		const std::vector<NeigbStr> &f_neigbs = mg_NeigbOfFluids[k];
		real_t rho0 = m_Fluids[k].restDensity_rho0;
		real_t fm0 = rho0 * v0;
		real_t wm0 = ker_W(0) * fm0;
		real_t h = 0.2;
		int num = int(f_parts.size());
		//int num_timer = 0;
		// first loop for foreach particle
#pragma omp parallel for
		for (int i = 0; i < num; ++i)
		{
			vec_t tempValue1 = vec_t::O;
			vec_t grad = vec_t::O;
			vec_t ni = vec_t::O;

			FluidPart &p_a = f_parts[i];
			real_t density = wm0;
			const Neigb *neigbs = f_neigbs[i].neigs;
			int n = f_neigbs[i].num;
			// forearch neighbour for computing rho
			for (int j = 0; j < n; ++j)
			{
				if (neigbs[j].pidx.isFluid())
				{ // fluid neighbour
					density += fm0 * ker_W(neigbs[j].dis);
				}
				else
				{ // boundary neighbour
					const BoundPart &p_b = getBoundPartOfIdx(neigbs[j].pidx);
					density += rho0 * p_b.volume * ker_W(neigbs[j].dis);
				}
			}
			p_a.density = density;

			// forearch neighbour for computing dii
			for (int j = 0; j < n; ++j)
			{
				grad = vec_t::O;
				if (neigbs[j].pidx.isFluid())
				{ // fluid neighbour
					const FluidPart &p_b = getFluidPartOfIdx(neigbs[j].pidx);
					grad = (p_a.position - p_b.position) * (-ker_W_grad(neigbs[j].dis) / neigbs[j].dis);
					tempValue1 += grad * (-fm0 / (p_a.density * p_a.density));

					//compute ni(��������)
					ni += grad * (fm0 / p_b.density);
				}
				else
				{ // boundary neighbour
					const BoundPart &p_b = getBoundPartOfIdx(neigbs[j].pidx);
					grad = (p_a.position - p_b.position) * (-ker_W_grad(neigbs[j].dis) / neigbs[j].dis);
					tempValue1 += grad * (-(rho0 * p_b.volume) / (p_a.density * p_a.density));
				}
			}
			p_a.dii = tempValue1 * m_TH.dt * m_TH.dt;
			ni = ni * m_TH.h;
			p_a.n = ni;

			//compute vel_adv
			p_a.vel_adv = p_a.velocity + p_a.acce_adv * m_TH.dt;
		}

		// second loop for foreach particle
#pragma omp parallel for
		for (int i = 0; i < num; ++i)
		{
			real_t tempValue1 = 0;
			real_t tempValue2 = 0;
			vec_t grad = vec_t::O;
			vec_t dji = vec_t::O;

			FluidPart &p_a = f_parts[i];
			const Neigb *neigbs = f_neigbs[i].neigs;
			int n = f_neigbs[i].num;
			// forearch neighbour for computing rho_adv,aii
			for (int j = 0; j < n; ++j)
			{
				grad = vec_t::O;
				dji = vec_t::O;
				if (neigbs[j].pidx.isFluid())
				{ // fluid neighbour
					const FluidPart &p_b = getFluidPartOfIdx(neigbs[j].pidx);
					grad = (p_a.position - p_b.position) * (-ker_W_grad(neigbs[j].dis) / neigbs[j].dis);
					//compute rho_adv
					tempValue1 += grad.dot((p_a.vel_adv - p_b.vel_adv)) * fm0 * m_TH.dt;
					//compute aii
					dji = grad * (fm0 * m_TH.dt * m_TH.dt / (p_a.density * p_a.density));
					tempValue2 += (p_a.dii - dji).dot(grad) * fm0;
				}
				else
				{ // boundary neighbour
					const BoundPart &p_b = getBoundPartOfIdx(neigbs[j].pidx);
					grad = (p_a.position - p_b.position) * (-ker_W_grad(neigbs[j].dis) / neigbs[j].dis);
					//compute rho_adv
					tempValue1 += grad.dot((p_a.velocity - p_b.velocity)) * (rho0 * p_b.volume) * m_TH.dt;
					//compute aii
					tempValue2 += (p_a.dii).dot(grad) * (rho0 * p_b.volume);
				}
			}
			if (tempValue2 == 0)
			{
				tempValue2 = 1;
			}
			p_a.aii = tempValue2;
			p_a.rho_adv = p_a.density + tempValue1;
			p_a.p_l = (0.5 * p_a.presure);
		}

		//pressure slover
		int l = 0;
		real_t averageDensityError = 1e10;
		real_t sumDensities = 0;
		long nbParticlesInSummation = 0;
		double eta = 0.001 * rho0;
		while ((averageDensityError > eta) || l < 2)
		{

			// third loop for foreach particle
#pragma omp parallel for
			for (int i = 0; i < num; ++i)
			{
				vec_t tempValue1 = vec_t::O;
				vec_t grad = vec_t::O;

				FluidPart &p_a = f_parts[i];
				const Neigb *neigbs = f_neigbs[i].neigs;
				int n = f_neigbs[i].num;
				// forearch neighbour
				for (int j = 0; j < n; ++j)
				{
					grad = vec_t::O;
					if (neigbs[j].pidx.isFluid())
					{ // fluid neighbour
						const FluidPart &p_b = getFluidPartOfIdx(neigbs[j].pidx);
						grad = (p_a.position - p_b.position) * (-ker_W_grad(neigbs[j].dis) / neigbs[j].dis);
						tempValue1 += grad * (-(fm0 * p_b.p_l) / (p_b.density * p_b.density));
					}
					else
					{ // boundary neighbour
						const BoundPart &p_b = getBoundPartOfIdx(neigbs[j].pidx);
						//??????????
					}
				}
				p_a.sum_dijpj = tempValue1 * (m_TH.dt * m_TH.dt);
			}

			// fourth loop for foreach particle
#pragma omp parallel for
			for (int i = 0; i < num; ++i)
			{
				vec_t grad = vec_t::O;
				vec_t tempValue1 = vec_t::O;
				real_t finalterm = 0;
				real_t t = 0;

				FluidPart &p_a = f_parts[i];
				const Neigb *neigbs = f_neigbs[i].neigs;
				int n = f_neigbs[i].num;
				// forearch neighbour
				for (int j = 0; j < n; ++j)
				{
					vec_t dji = vec_t::O;
					grad = vec_t::O;
					if (neigbs[j].pidx.isFluid())
					{ // fluid neighbour
						const FluidPart &p_b = getFluidPartOfIdx(neigbs[j].pidx);
						//compute dji
						grad = (p_a.position - p_b.position) * (-ker_W_grad(neigbs[j].dis) / neigbs[j].dis);
						dji = grad * m_TH.dt * m_TH.dt * fm0 / (p_a.density * p_a.density);

						// compute sum(djk*pk)-dji*pi
						tempValue1 = (p_a.sum_dijpj - p_b.dii * p_b.p_l) - (p_b.sum_dijpj - dji * p_a.p_l);
						finalterm = finalterm + (fm0 * tempValue1.dot(grad));
					}
					else
					{ // boundary neighbour
						const BoundPart &p_b = getBoundPartOfIdx(neigbs[j].pidx);
						grad = (p_a.position - p_b.position) * (-ker_W_grad(neigbs[j].dis) / neigbs[j].dis);
						finalterm = finalterm + (rho0 * p_b.volume * (p_a.sum_dijpj).dot(grad));
					}
				}
				t = 0.5 * p_a.p_l + (0.5 / p_a.aii) * (rho0 - p_a.rho_adv - finalterm);
				if (t < 0)
				{
					t = 0;
				}
				else
				{
					sumDensities += p_a.rho_adv + p_a.aii * p_a.p_l + finalterm;
					++nbParticlesInSummation;
				}
				p_a.p_l = t;
				p_a.presure = t;
				l = l + 1;
			}
			if (nbParticlesInSummation > 0)
			{
				averageDensityError = (sumDensities / num) - rho0;
			}
			else
			{
				averageDensityError = 0.0;
			}
		}
	}
}

// paritlce-particle interaction, [Mon92], [BT07]
inline void DFSPH::ii_fluidPartForceExceptPressure_fsame(
	FluidPart &fa, const FluidPart &fb, const real_t &dis,
	const real_t &fm0, const real_t &alpha, const real_t &gamma)
{
	if (dis == 0)
	{
		++m_EC.zeroDis;
		return;
	}
	vec_t xab = fa.position - fb.position;
	real_t grad = -ker_W_grad(dis) / dis;
	vec_t grad2 = (fa.position - fb.position) * (-ker_W_grad(dis) / dis);
	real_t acce = 0;
	// viscosity
	vec_t tmp = vec_t(0, 0, 0);

	real_t pro = (fa.velocity - fb.velocity).dot(xab);
	if (pro < 0)
	{
		tmp = grad2 * pro / (dis * dis + real_t(0.01) * m_TH.smoothRadius_h * m_TH.smoothRadius_h) * fm0 / fb.density * 10 * m_TH.nu;
	}

	/*if (m_TH.enable_vortex == 1) {
		fa.vis_acce += tmp;
	}*/
	fa.acce_adv += tmp;

	// surface tension
	// ...
}

// multiphase fluid, [SP08]
inline void DFSPH::ii_fluidPartForceExceptPressure_fdiff(
	FluidPart &fa, const FluidPart &fb, const real_t &dis,
	const real_t &fma, const real_t &fmb, const real_t &alpha)
{
	if (dis == 0)
	{
		++m_EC.zeroDis;
		return;
	}
	vec_t xab = fa.position - fb.position;
	real_t grad = -ker_W_grad(dis) / dis;
	real_t acce = 0;
	// viscosity
	real_t pro = (fa.velocity - fb.velocity).dot(xab);
	if (pro < 0)
	{
		real_t nu = 2 * alpha * m_TH.smoothRadius_h * m_TH.soundSpeed_cs / (fa.density + fb.density);
		real_t pi = -nu * pro / (dis * dis + real_t(0.01) * m_TH.smoothRadius_h * m_TH.smoothRadius_h);
		acce += grad * (-(fma + fmb) / 2 * pi);
	}
	if (acce)
	{
		xab *= acce;
		fa.acce_adv += xab;
	}
}

// fluid-rigid coupling, [AIS*12]
inline void DFSPH::ii_fluidPartForceExceptPressure_bound(
	FluidPart &fa, const BoundPart &rb, const real_t &dis,
	const real_t &frho0, const real_t &r_alpha)
{
	if (dis == 0)
	{
		++m_EC.zeroDis;
		return;
	}
	vec_t xab = fa.position - rb.position;
	real_t grad = -ker_W_grad(dis) / dis;
	vec_t grad2 = (fa.position - rb.position) * (-ker_W_grad(dis) / dis);
	real_t acce = 0;
	// viscosity
	real_t pro = (fa.velocity - rb.singleVelocity).dot(xab);
	//vec_t tmp = grad2 * pro / (dis*dis + real_t(0.01)*m_TH.smoothRadius_h*m_TH.smoothRadius_h) * (frho0*rb.volume) / fa.density * 10 * m_TH.nu;
	vec_t tmp = vec_t(0, 0, 0);
	if (pro < 0)
	{
		tmp = grad2 * pro / (dis * dis + real_t(0.01) * m_TH.smoothRadius_h * m_TH.smoothRadius_h) * (frho0 * rb.volume) / fa.density * 10 * m_TH.nu;
	}
	/*if (m_TH.enable_vortex == 1) {
		fa.vis_acce += tmp;
	}*/
	fa.acce_adv += tmp;
}

// Dynamic rigid step 2: 用该函数取代 ii_fluidPartForceExceptPressure_bound() 在 DF_forceExceptPressure() 中的地位
inline void DFSPH::DF_fluidBoundExceptPressure(
	FluidPart &fa, BoundPart &rb, const real_t &dis,
	const real_t &frho0, const real_t &r_alpha)
{
	if (dis == 0)
	{
		++m_EC.zeroDis;
		return;
	}
	vec_t xab = fa.position - rb.position;
	real_t grad = -ker_W_grad(dis) / dis;
	vec_t grad2 = (fa.position - rb.position) * (-ker_W_grad(dis) / dis);
	real_t acce = 0;

	// viscosity
	real_t pro = (fa.velocity - rb.singleVelocity - rb.velocity).dot(xab); // Dynamic Rigid
	vec_t tmp = vec_t(0, 0, 0);
	if (pro < 0)
	{
		tmp = grad2 * pro / (dis * dis + real_t(0.01) * m_TH.smoothRadius_h * m_TH.smoothRadius_h) * (frho0 * rb.volume) / fa.density * 10 * m_TH.nu;
	}

	fa.acce_adv += tmp;

	rb.force -= tmp;
}

void DFSPH::ii_forceExceptPressure()
{
	real_t v0 = std::pow(m_TH.spacing_r, vec_t::dim);
	// foreach fluid
	for (int n_f = int(m_Fluids.size()), k = 0; k < n_f; ++k)
	{
		std::vector<FluidPart> &f_parts = m_Fluids[k].fluidParticles;
		const std::vector<NeigbStr> &f_neigbs = mg_NeigbOfFluids[k];
		real_t rho0 = m_Fluids[k].restDensity_rho0;
		real_t fm0 = rho0 * v0;
		real_t alpha = m_Fluids[k].viscosity_alpha;
		real_t gamma = m_Fluids[k].surfaceTension_gamma;
		int num = int(f_parts.size());
		// foreach particle
#pragma omp parallel for
		for (int i = 0; i < num; ++i)
		{
			FluidPart &p_a = f_parts[i];
			p_a.presure = 0;
			p_a.acce_adv = m_TH.gravity_g;
			//p_a.acce_adv = vec_t::O;
			const Neigb *neigbs = f_neigbs[i].neigs;
			int n = f_neigbs[i].num;
			// forearch neighbour
			for (int j = 0; j < n; ++j)
			{
				if (neigbs[j].pidx.isFluid())
				{ // fluid neighbour
					const FluidPart &p_b = getFluidPartOfIdx(neigbs[j].pidx);
					int idx_b = neigbs[j].pidx.toFluidI();
					if (idx_b == k)
					{
						// the same fluid
						ii_fluidPartForceExceptPressure_fsame(
							p_a, p_b, neigbs[j].dis, fm0, alpha, gamma);
					}
					else
					{
						// different fluid
						real_t fmb = v0 * m_Fluids[idx_b].restDensity_rho0;
						real_t b_alpha = m_Fluids[idx_b].viscosity_alpha;
						ii_fluidPartForceExceptPressure_fdiff(
							p_a, p_b, neigbs[j].dis, fm0, fmb, (alpha + b_alpha) / 2);
					}
				}
				else
				{ // boundary neighbour
					const BoundPart &p_b = getBoundPartOfIdx(neigbs[j].pidx);
					int idx_b = neigbs[j].pidx.toSolidI();
					real_t r_alpha = m_Solids[idx_b].viscosity_alpha;
					ii_fluidPartForceExceptPressure_bound(
						p_a, p_b, neigbs[j].dis, rho0, r_alpha);
				}
			}
		}
	}
}

void DFSPH::DF_forceExceptPressure()
{
	real_t v0 = std::pow(m_TH.spacing_r, vec_t::dim);
	// foreach fluid
	for (int n_f = int(m_Fluids.size()), k = 0; k < n_f; ++k)
	{
		std::vector<FluidPart> &f_parts = m_Fluids[k].fluidParticles;
		const std::vector<NeigbStr> &f_neigbs = mg_NeigbOfFluids[k];
		real_t rho0 = m_Fluids[k].restDensity_rho0;
		real_t fm0 = rho0 * v0;
		real_t alpha = m_Fluids[k].viscosity_alpha;
		real_t gamma = m_Fluids[k].surfaceTension_gamma;
		int num = int(f_parts.size());
		// foreach particle
#pragma omp parallel for
		for (int i = 0; i < num; ++i)
		{
			FluidPart &p_a = f_parts[i];
			p_a.presure = 0;
			p_a.acce_adv = m_TH.gravity_g;
			//p_a.acce_adv = vec_t::O;
			const Neigb *neigbs = f_neigbs[i].neigs;
			int n = f_neigbs[i].num;
			// forearch neighbour
			for (int j = 0; j < n; ++j)
			{
				if (neigbs[j].pidx.isFluid())
				{ // fluid neighbour
					const FluidPart &p_b = getFluidPartOfIdx(neigbs[j].pidx);
					int idx_b = neigbs[j].pidx.toFluidI();
					if (idx_b == k)
					{
						// the same fluid
						ii_fluidPartForceExceptPressure_fsame(
							p_a, p_b, neigbs[j].dis, fm0, alpha, gamma);
					}
					else
					{
						// different fluid
						real_t fmb = v0 * m_Fluids[idx_b].restDensity_rho0;
						real_t b_alpha = m_Fluids[idx_b].viscosity_alpha;
						ii_fluidPartForceExceptPressure_fdiff(
							p_a, p_b, neigbs[j].dis, fm0, fmb, (alpha + b_alpha) / 2);
					}
				}
				else
				{ // boundary neighbour
					BoundPart &p_b = m_Solids[neigbs[j].pidx.toSolidI()].boundaryParticles[neigbs[j].pidx.i];
					int idx_b = neigbs[j].pidx.toSolidI();
					real_t r_alpha = m_Solids[idx_b].viscosity_alpha;
					DF_fluidBoundExceptPressure(
						p_a, p_b, neigbs[j].dis, rho0, r_alpha);
				}
			}
		}
	}
}

// paritlce-particle interaction, [Mon92], [BT07]
inline void DFSPH::ii_fluidPartForcePressure_fsame(
	FluidPart &fa, const FluidPart &fb, const real_t &dis,
	const real_t &fm0, const real_t &alpha, const real_t &gamma)
{
	if (dis == 0)
	{
		++m_EC.zeroDis;
		return;
	}
	vec_t xab = fa.position - fb.position;
	real_t grad = -ker_W_grad(dis) / dis;
	// momentum
	real_t acce = grad * (-fm0 * (fa.presure / (fa.density * fa.density) + fb.presure / (fb.density * fb.density)));
	// viscosity
	/*real_t pro = (fa.velocity-fb.velocity).dot( xab );
	if( pro<0 ){
	real_t nu = 2*alpha*m_TH.smoothRadius_h*m_TH.soundSpeed_cs / ( fa.density + fb.density );
	real_t pi =
	-nu * pro / ( dis*dis + real_t(0.01)*m_TH.smoothRadius_h*m_TH.smoothRadius_h );
	acce += grad * ( - fm0 * pi );
	}*/
	xab *= acce;
	fa.acce_presure += xab;

	// surface tension
	//real_t h = 0.2;
	//real_t r = 1;
	////compute cohesion
	//vec_t xab_coh = fa.position-fb.position;
	//double pai = 3.14;
	//real_t Kij = 2*1000/(fa.density+fb.density);
	//real_t C = 0;
	//if(2*dis>m_TH.h && dis<=m_TH.h){
	//	C = (32/(pai*pow(m_TH.h,9)))*pow(m_TH.h-dis,3)*pow(dis,3);
	//}else if(dis>0 && 2*dis<=m_TH.h){
	//	C = (32/(pai*pow(m_TH.h,9)))*(2*pow(m_TH.h-dis,3)*pow(dis,3)-pow(m_TH.h,6)/64);
	//}else{
	//	C = 0;
	//}
	//real_t acce_coh = -r*fm0*C/dis;
	//xab_coh *= acce_coh;

	////compute curvature
	//real_t acce_cur = -m_TH.r;
	//vec_t xab_cur = fa.n-fb.n;
	//xab_cur *= acce_cur;

	//vec_t acce_st = (xab_coh+xab_cur)*Kij;
	//fa.acce_presure += acce_st;
}
// multiphase fluid, [SP08]
inline void DFSPH::ii_fluidPartForcePressure_fdiff(
	FluidPart &fa, const FluidPart &fb, const real_t &dis,
	const real_t &fma, const real_t &fmb, const real_t &alpha)
{
	if (dis == 0)
	{
		++m_EC.zeroDis;
		return;
	}
	vec_t xab = fa.position - fb.position;
	real_t grad = -ker_W_grad(dis) / dis;
	real_t dalta_a = fa.density / fma, dalta_b = fb.density / fmb;
	// momentum
	real_t acce = grad * (-1 / fma * (fa.presure / (dalta_a * dalta_a) + fb.presure / (dalta_b * dalta_b)));
	// viscosity
	real_t pro = (fa.velocity - fb.velocity).dot(xab);
	if (pro < 0)
	{
		real_t nu = 2 * alpha * m_TH.smoothRadius_h * m_TH.soundSpeed_cs / (fa.density + fb.density);
		real_t pi =
			-nu * pro / (dis * dis + real_t(0.01) * m_TH.smoothRadius_h * m_TH.smoothRadius_h);
		acce += grad * (-(fma + fmb) / 2 * pi);
	}
	xab *= acce;
	fa.acce_presure += xab;
}
// fluid-rigid coupling, [AIS*12]
inline void DFSPH::ii_fluidPartForcePressure_bound(
	FluidPart &fa, const BoundPart &rb, const real_t &dis,
	const real_t &frho0, const real_t &r_alpha)
{
	if (dis == 0)
	{
		++m_EC.zeroDis;
		return;
	}
	vec_t xab = fa.position - rb.position;
	real_t grad = -ker_W_grad(dis) / dis, acce = 0;
	// momentum
	if (fa.presure > 0)
		acce += grad * (-frho0 * rb.volume * (fa.presure / (fa.density * fa.density) * 2));
	// viscosity
	/*real_t pro = (fa.velocity-rb.velocity).dot( xab );
	if( pro<0 ){
	real_t nu = 2*r_alpha*m_TH.smoothRadius_h*m_TH.soundSpeed_cs / ( fa.density*2 );
	real_t pi =
	-nu * pro / ( dis*dis + real_t(0.01)*m_TH.smoothRadius_h*m_TH.smoothRadius_h );
	acce += grad * (- frho0*rb.volume * pi );
	}*/
	xab *= acce;
	fa.acce_presure += xab;

	//// surface tension & adhesion
	//real_t bt = 1;
	//real_t h = 0.2;
	//real_t A = 0;
	//vec_t xab_adh = fa.position - rb.position;

	//if(2*dis>m_TH.h && dis<=m_TH.h){
	//	A = 0.007/(pow(m_TH.h,13/4))*pow((-4*dis*dis/m_TH.h+6*dis-2*m_TH.h),1/4);
	//}else{
	//	A = 0;
	//}
	//real_t acce_adh = -m_TH.bt*frho0*rb.volume*A/dis;
	//xab_adh *= acce_adh;
	//fa.acce_presure += xab_adh;
}
// fluid-rigid coupling, [AIS*12]
inline void DFSPH::ii_boundPartForcePressure_f(
	BoundPart &ra, const FluidPart &fb, const real_t &dis,
	const real_t &frho0, const real_t &fm0, const real_t &r_alpha)
{
	if (dis == 0)
	{
		++m_EC.zeroDis;
		return;
	}

	vec_t xab = ra.position - fb.position;
	real_t grad = -ker_W_grad(dis) / dis, force = 0;
	// momentum
	if (fb.presure > 0)
		force += grad * (-fm0 * frho0 * ra.volume * (fb.presure / (fb.density * fb.density) * 2));
	// viscosity
	real_t pro = (ra.velocity - fb.velocity).dot(xab);
	if (pro < 0)
	{
		real_t nu = 2 * r_alpha * m_TH.smoothRadius_h * m_TH.soundSpeed_cs / (fb.density * 2);
		real_t pi =
			-nu * pro / (dis * dis + real_t(0.01) * m_TH.smoothRadius_h * m_TH.smoothRadius_h);
		force += grad * (-fm0 * frho0 * ra.volume * pi);
	}
	xab *= force;
	ra.force += xab;
}
// compute gravity, pressure and friction force
void DFSPH::ii_computeForcePressure()
{
	real_t v0 = std::pow(m_TH.spacing_r, vec_t::dim);
	// foreach fluid
	for (int n_f = int(m_Fluids.size()), k = 0; k < n_f; ++k)
	{
		std::vector<FluidPart> &f_parts = m_Fluids[k].fluidParticles;
		const std::vector<NeigbStr> &f_neigbs = mg_NeigbOfFluids[k];
		real_t rho0 = m_Fluids[k].restDensity_rho0;
		real_t fm0 = rho0 * v0;
		real_t alpha = m_Fluids[k].viscosity_alpha;
		real_t gamma = m_Fluids[k].surfaceTension_gamma;
		int num = int(f_parts.size());
		// foreach particle
#pragma omp parallel for
		for (int i = 0; i < num; ++i)
		{
			FluidPart &p_a = f_parts[i];
			p_a.acce_presure = vec_t::O;
			const Neigb *neigbs = f_neigbs[i].neigs;
			int n = f_neigbs[i].num;
			// forearch neighbour
			for (int j = 0; j < n; ++j)
			{
				if (neigbs[j].pidx.isFluid())
				{ // fluid neighbour
					const FluidPart &p_b = getFluidPartOfIdx(neigbs[j].pidx);
					int idx_b = neigbs[j].pidx.toFluidI();
					// the same fluid
					if (idx_b == k)
					{
						ii_fluidPartForcePressure_fsame(p_a, p_b, neigbs[j].dis, fm0, alpha, gamma);
						// different fluid
					}
					else
					{
						real_t fmb = v0 * m_Fluids[idx_b].restDensity_rho0;
						real_t b_alpha = m_Fluids[idx_b].viscosity_alpha;
						ii_fluidPartForcePressure_fdiff(
							p_a, p_b, neigbs[j].dis, fm0, fmb, (alpha + b_alpha) / 2);
					}
				}
				else
				{ // boundary neighbour
					const BoundPart &p_b = getBoundPartOfIdx(neigbs[j].pidx);
					int idx_b = neigbs[j].pidx.toSolidI();
					real_t r_alpha = m_Solids[idx_b].viscosity_alpha;
					ii_fluidPartForcePressure_bound(p_a, p_b, neigbs[j].dis, rho0, r_alpha);
				}
			} //nneighbour
#ifdef II_ADT
			// the h in the paper is r, and the smoothing_h is 2h
			real_t v = !(p_a.velocity == vec_t::O)
						   ? m_Lambda_v * (m_TH.spacing_r / p_a.velocity.length())
						   : 1;
			real_t f = !(p_a.acceleration == vec_t::O)
						   ? m_Lambda_f * std::sqrt(m_TH.spacing_r / p_a.acceleration.length())
						   : 1;
			p_a.dt = std::min(v, f);
#endif
		} //particle
	}	  //fluid

	// forearch rigid
	for (int n_r = int(m_Solids.size()), k = 0; k < n_r; ++k)
		if (m_Solids[k].dynamic)
		{
			std::vector<BoundPart> &r_parts = m_Solids[k].boundaryParticles;
			const std::vector<NeigbStr> &r_neigbs = mg_NeigbOfSolids[k];
			real_t alpha = m_Solids[k].viscosity_alpha;
			int num = int(r_parts.size());
			// foreach particle
#pragma omp parallel for
			for (int i = 0; i < num; ++i)
			{
				BoundPart &p_a = r_parts[i];
				p_a.force = vec_t::O;
				const Neigb *neigbs = r_neigbs[i].neigs;
				int n = r_neigbs[i].num;
				// forearch neighbour
				for (int j = 0; j < n; ++j)
				{
					if (neigbs[j].pidx.isFluid())
					{ // fluid neighbour
						const FluidPart &p_b = getFluidPartOfIdx(neigbs[j].pidx);
						int idx_b = neigbs[j].pidx.toFluidI();
						real_t frho0 = m_Fluids[idx_b].restDensity_rho0;
						ii_boundPartForcePressure_f(p_a, p_b, neigbs[j].dis, frho0, frho0 * v0, alpha);
					}
				} //neighbour
			}	  //particle
		}		  //rigid
}

inline void DFSPH::fluidToSolid(int fluidIdx, const char *meshFileName,
								const char *sampleFileName, real_t viscosity, const glm::mat4 &transform)
{
}