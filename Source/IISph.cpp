
#define _SILENCE_STDEXT_HASH_DEPRECATION_WARNINGS 1

#include "IISph.h"
#include <iomanip>
#include "iostream"
#include <fstream>

using namespace std;
ofstream outfile;
int flag = 0;
// override SphBase::runOneStep()
void IISph::sphStep()
{
#define TIME_RECORD

#ifdef TIME_RECORD
	if (getFrameNumber() == 0) {
		m_clog.setf(std::ios::left);
#ifdef II_TIMEADAPTIVE
		m_clog << "Number\tTime\t\tselectActiv\tSearch\t\tWeight\t\tPressure\tForce\t\tconstrainDt\tadaptiveDt\tupFluid\t\tupRigid\t\tDT\t\t\tpercentOfActive\n";
#else
#ifdef II_ADT
		m_clog << "Number\tTime\tFluidNums\tCandidateNums\tSearch\t\tWeight\tPressure\tForce\tadaptiveDt\tupFluid\tupRigid\tDT\tdF_iteration\tcV_iteration\tpotential_energy\tkinetic_energy\tsum_energy\n";
#else
		m_clog << "Number\tTime\t\tSearch\t\tWeight\t\tPressure\tForce\t\tupFluid\t\tupRigid\t\tDT\n";
#endif
#endif
	}
	m_clog.width(7); m_clog << m_TH.frameNumber << ' ';
	m_clog.width(11); m_clog << m_TH.systemTime << ' ';

	m_clog << getNumFluidParts() << ' ';
	m_clog << getNumCandidateParts() << ' ';

	double t1, t2;
	
#define CALL_TIME(a) t1=omp_get_wtime(); a; t2=omp_get_wtime(); \
						 m_clog.width(11); m_clog <<  (t2-t1)*1000 << ' '
#else
#define CALL_TIME(a) a
#endif

#ifdef II_TIMEADAPTIVE
	CALL_TIME(selectActiveii());
#endif

	
	CALL_TIME(neighbourSearch());

	CALL_TIME(updateSolidPartWeight());

	//CALL_TIME(ii_computePressure());
	//CALL_TIME(ii_computeForcePressure());

	CALL_TIME(DF_prepareAttribute());
	//CALL_TIME(DF_prepareAttribute2());
	//CALL_TIME(DF_divergenceFreeSolver());
	//CALL_TIME(DF_constantDensitySolver());

	CALL_TIME(VF_prepareAttribute());
	CALL_TIME(VF_divergenceFreeSolver());
	CALL_TIME(VF_constantVolumeSolver());

	
	CALL_TIME(updateFluids());
	CALL_TIME(updateSolids());
	CALL_TIME(updateCandidates());

	if (m_TH.phaseTransition) {
		CALL_TIME(updateTemperature());
	}

	m_clog.width(11); m_clog << m_TH.dt << '\t';

	if (m_TH.systemTime > flag * 0.5) {	
		m_clog << m_TH.dF_iteration << '\t';
		m_TH.dF_iteration = 0; 
		m_clog << m_TH.cV_iteration << '\t';
		m_TH.cV_iteration = 0;
		//m_clog << setiosflags(ios::fixed);
		//m_clog << setprecision(6) << m_TH.potential_energy << '\t';
		//m_clog << setprecision(6) << m_TH.kinetic_energy << '\t';
		flag++;
	}

	

#ifdef II_TIMEADAPTIVE
	m_clog.width(11); m_clog << wc_percentOfActive << ' ';
#endif

#ifdef TIME_RECORD
	m_clog << '\n';
#endif

}

void IISph::DF_prepareAttribute() {
	ii_forceExceptPressure();

	for (int n_f = int(m_Fluids.size()), k = 0; k < n_f; ++k) {
		std::vector<FluidPart>& f_parts = m_Fluids[k].fluidParticles;
		const std::vector<NeigbStr>& f_neigbs = mg_NeigbOfFluids[k];
		real_t rho0 = m_Fluids[k].restDensity_rho0;
		int num = int(f_parts.size());

#pragma omp parallel for
		for (int i = 0; i < num; ++i) {
			FluidPart& p_a = f_parts[i];
			p_a.volume = std::pow(p_a.d, vec_t::dim);
			p_a.rho0 = rho0 * p_a.beta;
			p_a.fm0 = p_a.rho0 * p_a.volume;
		}

#pragma omp parallel for
		for (int i = 0; i < num; ++i) {
			FluidPart& p_a = f_parts[i];

			vec_t grad = vec_t::O;
			real_t gradScalar = 0;
			real_t density = ker_W(0) * p_a.fm0;

			p_a.DFalpha = 0;
			p_a.DFalpha1 = vec_t(0, 0, 0);
			p_a.DFalpha2 = 0;

			const Neigb* neigbs = f_neigbs[i].neigs; int n = f_neigbs[i].num;

			for (int j = 0; j < n; ++j) {
				if (neigbs[j].pidx.isFluid()) {
					const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);

					if (m_TH.adjustDensity) { density += p_a.fm0 * ker_W(neigbs[j].dis); }
					else{ density += p_b.fm0 * ker_W(neigbs[j].dis); }

					gradScalar = -ker_W_grad(neigbs[j].dis);
					grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

					p_a.DFalpha1 += grad * p_b.fm0;
					p_a.DFalpha2 += pow(gradScalar * p_b.fm0, 2);
				}
				else if (neigbs[j].pidx.isCandidate()) {
					const CandidatePart& p_c = getCandidatePartOfIdx(neigbs[j].pidx);
					density += p_a.rho0 * p_c.volume * ker_W(neigbs[j].dis);

					gradScalar = -ker_W_grad(neigbs[j].dis);
					grad = (p_a.position - p_c.position) * (gradScalar / neigbs[j].dis);

					p_a.DFalpha1 += grad * p_a.rho0 * p_c.volume;
				}
				else{
					const BoundPart& p_b = getBoundPartOfIdx(neigbs[j].pidx);
					density += p_a.rho0 * p_b.volume * ker_W(neigbs[j].dis);

					gradScalar = -ker_W_grad(neigbs[j].dis);
					grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

					p_a.DFalpha1 += grad * m_TH.boundaryDensity * p_b.volume;
				}
			}
			p_a.density = density;
			p_a.DFalpha = p_a.DFalpha1.len_square() + p_a.DFalpha2;
			if (p_a.DFalpha < 1.0e-6) {
				p_a.DFalpha = 1.0e-6;
			}
			p_a.DFalpha = p_a.density / p_a.DFalpha;
		}

#pragma omp parallel for
		for (int i = 0; i < num; ++i) {
			FluidPart& p_a = f_parts[i];
			vec_t grad;
			vec_t ni = vec_t::O;
			const Neigb* neigbs = f_neigbs[i].neigs; int n = f_neigbs[i].num;
			for (int j = 0; j < n; ++j) {
				grad = vec_t::O;
				if (neigbs[j].pidx.isFluid()) { // fluid neighbour
					const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);
					grad = (p_a.position - p_b.position) * (-ker_W_grad(neigbs[j].dis) / neigbs[j].dis);
					ni += grad * (p_b.fm0 / p_b.density); // question
				}
			}
			p_a.n = ni * m_TH.h;
		}
	}
}

void IISph::DF_prepareAttribute2() {
	ii_forceExceptPressure();

	for (int n_f = int(m_Fluids.size()), k = 0; k < n_f; ++k) {
		std::vector<FluidPart>& f_parts = m_Fluids[k].fluidParticles;
		const std::vector<NeigbStr>& f_neigbs = mg_NeigbOfFluids[k];

		real_t rho0 = m_Fluids[k].restDensity_rho0;
		int num = int(f_parts.size());

#pragma omp parallel for
		for (int i = 0; i < num; ++i) {
			FluidPart& p_a = f_parts[i];
			p_a.volume = std::pow(p_a.d, vec_t::dim);
			p_a.rho0 = rho0 * p_a.beta;
			p_a.fm0 = p_a.rho0 * p_a.volume;
		}

#pragma omp parallel for
		for (int i = 0; i < num; ++i) {
			FluidPart& p_a = f_parts[i];

			vec_t grad = vec_t::O;
			real_t gradScalar = 0;
			real_t density = ker_W(0) * p_a.fm0;

			p_a.DFalpha = 0;
			p_a.DFalpha1 = vec_t(0, 0, 0);
			p_a.DFalpha2 = 0;

			const Neigb* neigbs = f_neigbs[i].neigs; int n = f_neigbs[i].num;

			for (int j = 0; j < n; ++j) {
				if (neigbs[j].pidx.isFluid()) {
					const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);

					if (m_TH.adjustDensity) { density += p_a.fm0 * ker_W(neigbs[j].dis); }
					else { density += p_b.fm0 * ker_W(neigbs[j].dis); }

					gradScalar = -ker_W_grad(neigbs[j].dis);
					grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

					p_a.DFalpha1 += grad * p_b.fm0;
					p_a.DFalpha2 += pow(gradScalar, 2) * p_a.fm0 * p_b.fm0;
				}
				else if (neigbs[j].pidx.isCandidate()) {
					const CandidatePart& p_c = getCandidatePartOfIdx(neigbs[j].pidx);
					density += p_a.rho0 * p_c.volume * ker_W(neigbs[j].dis);

					gradScalar = -ker_W_grad(neigbs[j].dis);
					grad = (p_a.position - p_c.position) * (gradScalar / neigbs[j].dis);

					p_a.DFalpha1 += grad * p_a.rho0 * p_c.volume;
				}
				else {
					const BoundPart& p_b = getBoundPartOfIdx(neigbs[j].pidx);
					density += p_a.rho0 * p_b.volume * ker_W(neigbs[j].dis);

					gradScalar = -ker_W_grad(neigbs[j].dis);
					grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

					p_a.DFalpha1 += grad * m_TH.boundaryDensity * p_b.volume;
				}
			}
			p_a.density = density;
			p_a.DFalpha = p_a.DFalpha1.len_square() + p_a.DFalpha2;
			if (p_a.DFalpha < 1.0e-6) {
				p_a.DFalpha = 1.0e-6;
			}
			p_a.DFalpha = p_a.density / p_a.DFalpha;
		}

#pragma omp parallel for
		for (int i = 0; i < num; ++i) {
			FluidPart& p_a = f_parts[i];
			vec_t grad;
			vec_t ni = vec_t::O;
			const Neigb* neigbs = f_neigbs[i].neigs; int n = f_neigbs[i].num;
			for (int j = 0; j < n; ++j) {
				grad = vec_t::O;
				if (neigbs[j].pidx.isFluid()) { // fluid neighbour
					const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);
					grad = (p_a.position - p_b.position) * (-ker_W_grad(neigbs[j].dis) / neigbs[j].dis);
					ni += grad * (p_b.fm0 / p_b.density); // question
				}
			}
			p_a.n = ni * m_TH.h;
		}
	}
}

void IISph::VF_prepareAttribute() {
	ii_forceExceptPressure();

	for (int n_f = int(m_Fluids.size()), k = 0; k < n_f; ++k) {

		std::vector<FluidPart>& f_parts = m_Fluids[k].fluidParticles;
		const std::vector<NeigbStr>& f_neigbs = mg_NeigbOfFluids[k];

		real_t rho0 = m_Fluids[k].restDensity_rho0;
		int num = int(f_parts.size());

#pragma omp parallel for
		for (int i = 0; i < num; ++i) {
			FluidPart& p_a = f_parts[i];
			p_a.volume = std::pow(p_a.d, vec_t::dim);
			p_a.rho0 = rho0 * p_a.beta;
			p_a.fm0 = p_a.rho0 * p_a.volume;
		}

#pragma omp parallel for
		for (int i = 0; i < num; ++i) {
			FluidPart& p_a = f_parts[i];

			vec_t grad = vec_t::O;
			real_t gradScalar = 0;
			real_t VFvolume = ker_W(0) * pow(p_a.volume, 2);

			p_a.VFvolume = 0;
			p_a.VFalpha = 0;
			p_a.VFalpha1 = vec_t(0, 0, 0);
			p_a.VFalpha2 = 0;
			p_a.VFalpha3 = 0;

			const Neigb* neigbs = f_neigbs[i].neigs; int n = f_neigbs[i].num;

			for (int j = 0; j < n; ++j) {

				if (neigbs[j].pidx.isFluid()) {
					const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);

					VFvolume += pow(p_b.volume, 2) * ker_W(neigbs[j].dis);

					gradScalar = -ker_W_grad(neigbs[j].dis);
					grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

					p_a.VFalpha1 += grad * pow(p_b.volume, 2);
					p_a.VFalpha2 += pow(p_b.volume, 4) / p_b.fm0 * grad.dot(grad);
				}
				else if (neigbs[j].pidx.isCandidate()) {
					const CandidatePart& p_c = getCandidatePartOfIdx(neigbs[j].pidx);

					VFvolume += pow(p_c.volume, 2) * ker_W(neigbs[j].dis);

					gradScalar = -ker_W_grad(neigbs[j].dis);
					grad = (p_a.position - p_c.position) * (gradScalar / neigbs[j].dis);

					p_a.VFalpha1 += grad * pow(p_c.volume, 2);
					p_a.VFalpha2 += pow(p_c.volume, 4) / p_a.fm0 * grad.dot(grad);
				}
				else {
					const BoundPart& p_b = getBoundPartOfIdx(neigbs[j].pidx);

					VFvolume += pow(p_b.volume, 2) * ker_W(neigbs[j].dis);

					gradScalar = -ker_W_grad(neigbs[j].dis);
					grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

					p_a.VFalpha1 += grad * pow(p_b.volume, 2);
					p_a.VFalpha2 += pow(p_b.volume, 4) / p_a.fm0 * grad.dot(grad);
				}
			}

			for (int j = 0; j < n; ++j) {

				if (neigbs[j].pidx.isFluid()) {
					const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);

					gradScalar = -ker_W_grad(neigbs[j].dis);
					grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

					p_a.VFalpha3 += grad.dot(p_a.VFalpha1 * pow(p_b.volume, 2) / p_a.fm0);
				}
				else if (neigbs[j].pidx.isCandidate()) {
					const CandidatePart& p_c = getCandidatePartOfIdx(neigbs[j].pidx);

					gradScalar = -ker_W_grad(neigbs[j].dis);
					grad = (p_a.position - p_c.position) * (gradScalar / neigbs[j].dis);

					p_a.VFalpha3 += grad.dot(p_a.VFalpha1 * pow(p_c.volume, 2) / p_a.fm0);
				}
				else {
					const BoundPart& p_b = getBoundPartOfIdx(neigbs[j].pidx);

					gradScalar = -ker_W_grad(neigbs[j].dis);
					grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

					p_a.VFalpha3 += grad.dot(p_a.VFalpha1 * pow(p_b.volume, 2) / p_a.fm0);
				}
			}

			p_a.VFvolume = VFvolume;
			p_a.VFalpha = p_a.VFalpha2 + p_a.VFalpha3;

			if (p_a.VFalpha < 1.0e-10) {
				p_a.VFalpha = 1.0e-10;
			}
			p_a.VFalpha = 1 / p_a.VFalpha;
		}

#pragma omp parallel for
		for (int i = 0; i < num; ++i) {
			FluidPart& p_a = f_parts[i];
			vec_t grad;
			vec_t ni = vec_t::O;
			const Neigb* neigbs = f_neigbs[i].neigs; int n = f_neigbs[i].num;
			for (int j = 0; j < n; ++j) {
				grad = vec_t::O;
				if (neigbs[j].pidx.isFluid()) { // fluid neighbour
					const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);
					grad = (p_a.position - p_b.position) * (-ker_W_grad(neigbs[j].dis) / neigbs[j].dis);
					ni += grad * (1 / p_b.VFvolume); // question
				}
			}
			p_a.n = ni * m_TH.h;
		}
	}
}

void IISph::VF_constantVolumeSolver() {

	ii_forceExceptPressure();

	for (int n_f = int(m_Fluids.size()), k = 0; k < n_f; ++k) {
		
		std::vector<FluidPart>& f_parts = m_Fluids[k].fluidParticles;
		const std::vector<NeigbStr>& f_neigbs = mg_NeigbOfFluids[k];

		int num = int(f_parts.size());

		real_t errorRateGoal = 0.0001;
		real_t errorRate = 1;
		real_t sumErrorRate = 0;
		int maximumIteration = 100;
		int minimumIteration = 3;
		int currentIteration = 0;

#pragma omp parallel for
		for (int i = 0; i < num; ++i) {
			FluidPart& p_a = f_parts[i];
			p_a.volume = pow(p_a.d, vec_t::dim);
			p_a.vel_adv = p_a.velocity + p_a.acce_adv * m_TH.dt;
		}

		while (errorRate > errorRateGoal || currentIteration < minimumIteration) {
			//cout << "constant volume iter: " << currentIteration << endl;
			if (currentIteration > maximumIteration) break;
			currentIteration++;

			real_t densitySum = 0;

#pragma omp parallel for
			for (int i = 0; i < num; ++i) {
				FluidPart& p_a = f_parts[i];

				vec_t grad = vec_t::O;
				real_t gradScalar = 0;

				p_a.VFvolume_adv = p_a.VFvolume;

				const Neigb* neigbs = f_neigbs[i].neigs; int n = f_neigbs[i].num;

				for (int j = 0; j < n; ++j) {
					if (neigbs[j].pidx.isFluid()) {
						const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);

						gradScalar = -ker_W_grad(neigbs[j].dis);
						grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

						p_a.VFvolume_adv += grad.dot(p_a.vel_adv - p_b.vel_adv) * pow(p_b.volume, 2) * m_TH.dt;
					}
					else if (neigbs[j].pidx.isCandidate()) {
						const CandidatePart& p_c = getCandidatePartOfIdx(neigbs[j].pidx);

						gradScalar = -ker_W_grad(neigbs[j].dis);
						grad = (p_a.position - p_c.position) * (gradScalar / neigbs[j].dis);

						p_a.VFvolume_adv += grad.dot(p_a.vel_adv - p_c.velocity) * pow(p_c.volume, 2) * m_TH.dt;
					}
					else {
						const BoundPart& p_b = getBoundPartOfIdx(neigbs[j].pidx);

						gradScalar = -ker_W_grad(neigbs[j].dis);
						grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

						p_a.VFvolume_adv += grad.dot(p_a.vel_adv - p_b.velocity) * pow(p_b.volume, 2) * m_TH.dt;
					}
				}

				if (p_a.VFvolume_adv < p_a.volume) {
					p_a.VFvolume_adv = p_a.volume;
				}
			}

#pragma omp parallel for
			for (int i = 0; i < num; ++i) { //particle "i" in "k"
				FluidPart& p_a = f_parts[i];
				p_a.VFkappa = (p_a.VFvolume_adv - p_a.volume) / pow(m_TH.dt, 2) * p_a.VFvolume / pow(p_a.volume, 2) * p_a.VFalpha;
			}
			
#pragma omp parallel for
			for (int i = 0; i < num; ++i) { //particle "i" in "k"
				FluidPart& p_a = f_parts[i];

				vec_t grad = vec_t::O;
				real_t gradScalar = 0;

				const Neigb* neigbs = f_neigbs[i].neigs; int n = f_neigbs[i].num;

				for (int j = 0; j < n; ++j) {
					if (neigbs[j].pidx.isFluid()) {
						const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);

						gradScalar = -ker_W_grad(neigbs[j].dis);
						grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

						p_a.vel_adv = p_a.vel_adv - (grad * m_TH.dt / p_a.fm0 * ((pow(p_a.volume, 4) * p_a.VFkappa / p_a.VFvolume) + (pow(p_b.volume, 4) * p_b.VFkappa / p_b.VFvolume)));
					}
					else if (neigbs[j].pidx.isCandidate()) {
						const CandidatePart& p_c = getCandidatePartOfIdx(neigbs[j].pidx);

						gradScalar = -ker_W_grad(neigbs[j].dis);
						grad = (p_a.position - p_c.position) * (gradScalar / neigbs[j].dis);

						p_a.vel_adv = p_a.vel_adv - (grad * m_TH.dt / p_a.fm0 * (pow(p_c.volume, 2) * pow(p_a.volume, 2) * p_a.VFkappa / p_a.VFvolume) * 2); // question
					}
					else {
						const BoundPart& p_b = getBoundPartOfIdx(neigbs[j].pidx);

						gradScalar = -ker_W_grad(neigbs[j].dis);
						grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

						p_a.vel_adv = p_a.vel_adv - (grad * m_TH.dt / p_a.fm0 * (pow(p_b.volume, 2) * pow(p_a.volume, 2) * p_a.VFkappa / p_a.VFvolume) * 2); // question
					}
				}

#pragma omp critical
				{sumErrorRate += (p_a.VFvolume_adv - p_a.volume) / p_a.volume;}
			}

			errorRate = sumErrorRate / num;
			//cout << "volume error rate: " << errorRate << endl;
		}
		m_TH.cV_iteration += currentIteration - 1;
#pragma omp parallel for
		for (int i = 0; i < num; ++i) {
			FluidPart& p_a = f_parts[i];
			p_a.acce_presure = (p_a.vel_adv - p_a.velocity) / m_TH.dt;
		}

		//计算energy
		if (m_TH.enable_energy_computation) {
			if (m_TH.systemTime > m_TH.energy_tracing_frequency* m_TH.frequency_timer) {
				m_TH.potential_energy = 0;
				m_TH.kinetic_energy = 0;
				m_TH.sum_energy = 0;
				for (int i = 0; i < num; ++i) {
					FluidPart& p_a = f_parts[i];
					m_TH.potential_energy += p_a.fm0 * -m_TH.gravity_g[1] * (p_a.position[1] + 5);  // total potential energy
					m_TH.kinetic_energy += 0.5 * p_a.fm0 * p_a.velocity.dot(p_a.velocity); //total kinetic energy
				}
				m_TH.sum_energy = m_TH.potential_energy + m_TH.kinetic_energy;
				m_clog << setiosflags(ios::fixed);
				m_clog << setprecision(6) << m_TH.potential_energy / num << '\t';
				m_clog << setprecision(6) << m_TH.kinetic_energy / num << '\t';
				m_clog << setprecision(6) << m_TH.sum_energy / num << '\t';
				outfile.open("energy.txt", ios::app);
				outfile << m_TH.sum_energy / num << "\t" << m_TH.potential_energy / num << "\t" << m_TH.kinetic_energy / num << "\n";
				outfile.close();
				m_TH.frequency_timer++;
			}
		}
	}
}

void IISph::DF_constantDensitySolver() {

	ii_forceExceptPressure();

	for (int n_f = int(m_Fluids.size()), k = 0; k < n_f; ++k) {

		std::vector<FluidPart>& f_parts = m_Fluids[k].fluidParticles;
		const std::vector<NeigbStr>& f_neigbs = mg_NeigbOfFluids[k];

		int num = int(f_parts.size());

		real_t errorRateGoal = 0.0001;
		real_t errorRate = 1;
		int maximumIteration = 100;
		int minimumIteration = 3;
		int currentIteration = 0;

#pragma omp parallel for
		for (int i = 0; i < num; ++i) {
			FluidPart& p_a = f_parts[i]; 
			p_a.volume = pow(p_a.d, vec_t::dim);
			p_a.vel_adv = p_a.velocity + p_a.acce_adv * m_TH.dt;
		}

		while (errorRate > errorRateGoal || currentIteration < minimumIteration) {
			//cout << "constant density iter: " << currentIteration << endl;
			if (currentIteration > maximumIteration) break;
			currentIteration++;

			real_t densitySum = 0;

#pragma omp parallel for
			for (int i = 0; i < num; ++i) {

				FluidPart& p_a = f_parts[i];

				vec_t grad = vec_t::O; //gradient
				real_t gradScalar = 0; //gradient scalar version
				p_a.density_adv = p_a.density;

				const Neigb* neigbs = f_neigbs[i].neigs; 
				int n = f_neigbs[i].num;

				for (int j = 0; j < n; ++j) {
					if (neigbs[j].pidx.isFluid()) {
						const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);

						gradScalar = -ker_W_grad(neigbs[j].dis);
						grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

						if (m_TH.adjustDensity) {
							p_a.density_adv += grad.dot(p_a.vel_adv - p_b.vel_adv) * p_a.fm0 * m_TH.dt;
						}
						else
						{
							p_a.density_adv += grad.dot(p_a.vel_adv - p_b.vel_adv) * p_b.fm0 * m_TH.dt;
						}
					}
					else if (neigbs[j].pidx.isCandidate()) {
						const CandidatePart& p_c = getCandidatePartOfIdx(neigbs[j].pidx);

						gradScalar = -ker_W_grad(neigbs[j].dis);
						grad = (p_a.position - p_c.position) * (gradScalar / neigbs[j].dis);

						p_a.density_adv += grad.dot(p_a.vel_adv - p_c.velocity) * p_a.rho0 * p_c.volume * m_TH.dt;
					}
					else {
						const BoundPart& p_b = getBoundPartOfIdx(neigbs[j].pidx);

						gradScalar = -ker_W_grad(neigbs[j].dis);
						grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

						p_a.density_adv += grad.dot(p_a.vel_adv - p_b.velocity) * m_TH.boundaryDensity * p_b.volume * m_TH.dt;
					}
				}

				if (p_a.density_adv < p_a.rho0) { // when lack of neighbours, p_a should be regarded as restDensity
					p_a.density_adv = p_a.rho0;
				}

#pragma omp critical
				{ densitySum += (p_a.density_adv - p_a.rho0) / p_a.rho0; }
			}

#pragma omp parallel for
			for (int i = 0; i < num; ++i) {
				FluidPart& p_a = f_parts[i];
				p_a.DFkappa = (p_a.density_adv - p_a.rho0) / pow(m_TH.dt, 2) * p_a.DFalpha;
			}

#pragma omp parallel for
			for (int i = 0; i < num; ++i) {
				FluidPart& p_a = f_parts[i];

				vec_t grad = vec_t::O;
				real_t gradScalar = 0;

				const Neigb* neigbs = f_neigbs[i].neigs; int n = f_neigbs[i].num;

				for (int j = 0; j < n; ++j) {
					if (neigbs[j].pidx.isFluid()) {
						const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);

						gradScalar = -ker_W_grad(neigbs[j].dis);
						grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

						p_a.vel_adv = p_a.vel_adv - (grad * p_b.fm0 * ((p_a.DFkappa / p_a.density) + (p_b.DFkappa / p_b.density)) * m_TH.dt);
					}
					else if (neigbs[j].pidx.isCandidate()) {
						const CandidatePart& p_c = getCandidatePartOfIdx(neigbs[j].pidx);

						gradScalar = -ker_W_grad(neigbs[j].dis);
						grad = (p_a.position - p_c.position) * (gradScalar / neigbs[j].dis);

						p_a.vel_adv = p_a.vel_adv - (grad * p_a.rho0 * p_c.volume * ((p_a.DFkappa / p_a.density)*2) * m_TH.dt);
					}
					else{
						const BoundPart& p_b = getBoundPartOfIdx(neigbs[j].pidx);

						gradScalar = -ker_W_grad(neigbs[j].dis);
						grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

						p_a.vel_adv = p_a.vel_adv - (grad * m_TH.boundaryDensity * p_b.volume * ((p_a.DFkappa / p_a.density)*2) * m_TH.dt);
					}
				}
			}

			errorRate = densitySum / num;
			//cout << "density error rate: " << errorRate << endl;
		}
		m_TH.cV_iteration += currentIteration - 1;
#pragma omp parallel for
		for (int i = 0; i < num; ++i) {
			FluidPart& p_a = f_parts[i];
			p_a.acce_presure = (p_a.vel_adv - p_a.velocity) / m_TH.dt;
		}

		//计算energy
		if (m_TH.enable_energy_computation) {
			if (m_TH.systemTime > m_TH.energy_tracing_frequency* m_TH.frequency_timer) {
				m_TH.potential_energy = 0;
				m_TH.kinetic_energy = 0;
				m_TH.sum_energy = 0;
				for (int i = 0; i < num; ++i) {
					FluidPart& p_a = f_parts[i];
					m_TH.potential_energy += p_a.fm0 * -m_TH.gravity_g[1] * (p_a.position[1] + 5);  // total potential energy
					m_TH.kinetic_energy += 0.5 * p_a.fm0 * p_a.velocity.dot(p_a.velocity); //total kinetic energy
				}
				m_TH.sum_energy = m_TH.potential_energy + m_TH.kinetic_energy;
				m_clog << setiosflags(ios::fixed);
				m_clog << setprecision(6) << m_TH.potential_energy / num << '\t';
				m_clog << setprecision(6) << m_TH.kinetic_energy / num << '\t';
				m_clog << setprecision(6) << m_TH.sum_energy / num << '\t';
				outfile.open("energy.txt", ios::app);
				outfile << m_TH.sum_energy / num << "\t" << m_TH.potential_energy / num << "\t" << m_TH.kinetic_energy / num << "\n";
				outfile.close();
				m_TH.frequency_timer++;
			}
		}
	}
}

void IISph::DF_divergenceFreeSolver() {

	for (int n_f = int(m_Fluids.size()), k = 0; k < n_f; ++k) {

		std::vector<FluidPart>& f_parts = m_Fluids[k].fluidParticles;
		const std::vector<NeigbStr>& f_neigbs = mg_NeigbOfFluids[k];

		int num = int(f_parts.size());

		real_t errorRateGoal = 0.001;
		real_t divergenceDeviationAver = 10;
		int maximumIteration = 30;
		int minimumIteration = 1;
		int currentIteration = 0;

#pragma omp parallel for
		for (int i = 0; i < num; ++i) {
			FluidPart& p_a = f_parts[i];
			p_a.volume = pow(p_a.d, vec_t::dim);
			p_a.vel_adv = p_a.velocity;
		}

		while (divergenceDeviationAver > errorRateGoal || currentIteration < minimumIteration) {
			//cout << "divergence free iter: " << currentIteration << endl;
			if (currentIteration > maximumIteration) break;
			currentIteration++;
			divergenceDeviationAver = 0;

#pragma omp parallel for
			for (int i = 0; i < num; ++i) {
				FluidPart& p_a = f_parts[i]; 

				vec_t grad = vec_t::O;
				real_t gradScalar = 0;
				p_a.DFdivergenceDeviation = 0;

				/* neighbour<vec> and its size */
				const Neigb* neigbs = f_neigbs[i].neigs; int n = f_neigbs[i].num;

				for (int j = 0; j < n; ++j) {
					if (neigbs[j].pidx.isFluid()) {
						const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);

						gradScalar = -ker_W_grad(neigbs[j].dis);
						grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

						p_a.DFdivergenceDeviation += grad.dot(p_a.vel_adv - p_b.vel_adv) * p_b.fm0;
					}
					else if (neigbs[j].pidx.isCandidate()) {
						const CandidatePart& p_c = getCandidatePartOfIdx(neigbs[j].pidx);

						gradScalar = -ker_W_grad(neigbs[j].dis);
						grad = (p_a.position - p_c.position) * (gradScalar / neigbs[j].dis);

						p_a.DFdivergenceDeviation += grad.dot(p_a.vel_adv - p_c.velocity) * (p_a.rho0 * p_c.volume);
					}
					else {
						const BoundPart& p_b = getBoundPartOfIdx(neigbs[j].pidx);

						gradScalar = -ker_W_grad(neigbs[j].dis);
						grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

						p_a.DFdivergenceDeviation += grad.dot(p_a.vel_adv - p_b.velocity) * (p_a.rho0 * p_b.volume);
					}
				}

				if (p_a.DFdivergenceDeviation < 0 || (p_a.density + p_a.DFdivergenceDeviation * m_TH.dt) < p_a.rho0) {
					p_a.DFdivergenceDeviation = 0;
				}

				p_a.DFkappaV = p_a.DFdivergenceDeviation * p_a.DFalpha / m_TH.dt;
			}

#pragma omp parallel for
			for (int i = 0; i < num; ++i) { 
				FluidPart& p_a = f_parts[i];

				vec_t grad = vec_t::O;
				real_t gradScalar = 0;

				const Neigb* neigbs = f_neigbs[i].neigs; int n = f_neigbs[i].num;

				for (int j = 0; j < n; ++j) {
					if (neigbs[j].pidx.isFluid()) {
						const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);

						gradScalar = -ker_W_grad(neigbs[j].dis);
						grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

						p_a.vel_adv = p_a.vel_adv - (grad * p_b.fm0 * ((p_a.DFkappaV / p_a.density) + (p_b.DFkappaV / p_b.density)) * m_TH.dt);
					}
					else if (neigbs[j].pidx.isCandidate()) {
						const CandidatePart& p_c = getCandidatePartOfIdx(neigbs[j].pidx);

						gradScalar = -ker_W_grad(neigbs[j].dis);
						grad = (p_a.position - p_c.position) * (gradScalar / neigbs[j].dis);

						p_a.vel_adv = p_a.vel_adv - (grad * p_a.rho0 * p_c.volume * ((p_a.DFkappaV / p_a.density)*2) * m_TH.dt);
					}
					else {
						const BoundPart& p_b = getBoundPartOfIdx(neigbs[j].pidx);

						gradScalar = -ker_W_grad(neigbs[j].dis);
						grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

						p_a.vel_adv = p_a.vel_adv - (grad * p_a.rho0 * p_b.volume * ((p_a.DFkappaV / p_a.density)*2) * m_TH.dt);
					}
				}

#pragma omp critical
				{ divergenceDeviationAver += p_a.DFdivergenceDeviation; }
			}
			divergenceDeviationAver = divergenceDeviationAver / num * m_TH.dt;
			//cout << "divergence correction: " << divergenceDeviationAver << endl;
		}
		m_TH.dF_iteration += currentIteration - 1;
#pragma omp parallel for
		for (int i = 0; i < num; ++i) {
			FluidPart& p_a = f_parts[i];
			p_a.velocity = p_a.vel_adv;
		}
	}
}

void IISph::VF_divergenceFreeSolver() {

	for (int n_f = int(m_Fluids.size()), k = 0; k < n_f; ++k) {

		std::vector<FluidPart>& f_parts = m_Fluids[k].fluidParticles;
		const std::vector<NeigbStr>& f_neigbs = mg_NeigbOfFluids[k];

		int num = int(f_parts.size());

		real_t errorRateGoal = 0.001;
		real_t divergenceDeviationAver = 10;
		int maximumIteration = 50;
		int minimumIteration = 2;
		int currentIteration = 0;

#pragma omp parallel for
		for (int i = 0; i < num; ++i) {
			FluidPart& p_a = f_parts[i];
			p_a.volume = pow(p_a.d, vec_t::dim);
			p_a.vel_adv = p_a.velocity;
		}

		while (divergenceDeviationAver > errorRateGoal || currentIteration < minimumIteration) {
			//cout << "divergence free iter: " << currentIteration << endl;
			if (currentIteration > maximumIteration) break;
			currentIteration++;
			divergenceDeviationAver = 0;

#pragma omp parallel for
			for (int i = 0; i < num; ++i) {

				FluidPart& p_a = f_parts[i];

				vec_t grad = vec_t::O;
				real_t gradScalar = 0;
				p_a.VFdivergenceDeviation = 0;

				const Neigb* neigbs = f_neigbs[i].neigs; int n = f_neigbs[i].num;

				for (int j = 0; j < n; ++j) {
					if (neigbs[j].pidx.isFluid()) {
						const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);

						gradScalar = -ker_W_grad(neigbs[j].dis);
						grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

						p_a.VFdivergenceDeviation += grad.dot(p_a.vel_adv - p_b.vel_adv) * pow(p_b.volume, 2);
					}
					else if (neigbs[j].pidx.isCandidate()) {
						const CandidatePart& p_c = getCandidatePartOfIdx(neigbs[j].pidx);

						gradScalar = -ker_W_grad(neigbs[j].dis);
						grad = (p_a.position - p_c.position) * (gradScalar / neigbs[j].dis);

						p_a.VFdivergenceDeviation += grad.dot(p_a.vel_adv - p_c.velocity) * pow(p_c.volume, 2);
					}
					else {
						const BoundPart& p_b = getBoundPartOfIdx(neigbs[j].pidx);

						gradScalar = -ker_W_grad(neigbs[j].dis);
						grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

						p_a.VFdivergenceDeviation += grad.dot(p_a.vel_adv - p_b.velocity) * pow(p_b.volume, 2);
					}
				}

				if (p_a.VFdivergenceDeviation < 0 || (p_a.VFvolume + p_a.VFdivergenceDeviation * m_TH.dt) < p_a.volume) {
					p_a.VFdivergenceDeviation = 0;
				}

				p_a.VFkappaV = p_a.VFdivergenceDeviation * p_a.VFalpha / m_TH.dt * p_a.VFvolume / pow(p_a.volume, 2);
			}

#pragma omp parallel for
			for (int i = 0; i < num; ++i) {

				FluidPart& p_a = f_parts[i];

				vec_t grad = vec_t::O;
				real_t gradScalar = 0;

				const Neigb* neigbs = f_neigbs[i].neigs; int n = f_neigbs[i].num;

				for (int j = 0; j < n; ++j) {
					if (neigbs[j].pidx.isFluid()) {
						const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);

						gradScalar = -ker_W_grad(neigbs[j].dis);
						grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

						p_a.vel_adv = p_a.vel_adv - (grad * m_TH.dt / p_a.fm0 * ((pow(p_a.volume, 4) * p_a.VFkappaV / p_a.VFvolume) + (pow(p_b.volume, 4) * p_b.VFkappaV / p_b.VFvolume)));
					}
					else if (neigbs[j].pidx.isCandidate()) {
						const CandidatePart& p_c = getCandidatePartOfIdx(neigbs[j].pidx);

						gradScalar = -ker_W_grad(neigbs[j].dis);
						grad = (p_a.position - p_c.position) * (gradScalar / neigbs[j].dis);

						p_a.vel_adv = p_a.vel_adv - (grad * m_TH.dt / p_a.fm0 * (pow(p_c.volume, 2) * pow(p_a.volume, 2) * p_a.VFkappaV / p_a.VFvolume) * 2);
					}
					else {
						const BoundPart& p_b = getBoundPartOfIdx(neigbs[j].pidx);

						gradScalar = -ker_W_grad(neigbs[j].dis);
						grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

						p_a.vel_adv = p_a.vel_adv - (grad * m_TH.dt / p_a.fm0 * (pow(p_b.volume, 2) * pow(p_a.volume, 2) * p_a.VFkappaV / p_a.VFvolume) * 2);
					}
				}
#pragma omp critical
				{ divergenceDeviationAver += p_a.VFdivergenceDeviation; }
			}

			divergenceDeviationAver = divergenceDeviationAver / num * m_TH.dt;
			//cout << "divergence error rate: " << divergenceDeviationAver << endl;
		}

		m_TH.dF_iteration += currentIteration - 1;
#pragma omp parallel for
		for (int i = 0; i < num; ++i) {
			FluidPart& p_a = f_parts[i];
			p_a.velocity = p_a.vel_adv;
		}
	}
}

// compute fluid particles' pressure, WCSPH or PCISPH
void IISph::ii_computePressure()
{
	//compute acce_adv;
	ii_forceExceptPressure();

	// foreach fluid
	for (int n_f = int(m_Fluids.size()), k = 0; k < n_f; ++k) {
		std::vector<FluidPart>& f_parts = m_Fluids[k].fluidParticles;
		const std::vector<NeigbStr>& f_neigbs = mg_NeigbOfFluids[k];
		real_t rho0_ori = m_Fluids[k].restDensity_rho0;
		//real_t fm0 = rho0 * v0;
		//real_t wm0 = ker_W(0) * fm0;
		real_t h = m_TH.h;
		int num = int(f_parts.size());
		// first loop for foreach particle 
#pragma omp parallel for
		for (int i = 0; i < num; ++i) {
			FluidPart& p_a = f_parts[i];
			p_a.volume = pow(p_a.d, vec_t::dim);
			p_a.rho0 = p_a.beta * rho0_ori;
			p_a.fm0 = p_a.rho0 * p_a.volume;
		}
#pragma omp parallel for
		for (int i = 0; i < num; ++i) {
			vec_t tempValue1 = vec_t::O;
			vec_t grad = vec_t::O;
			vec_t ni = vec_t::O;
			FluidPart& p_a = f_parts[i];
			real_t wm0 = ker_W(0) * p_a.fm0;

			real_t gamma = ker_W(0) * p_a.volume;
			//real_t gamma_temp = ker_W(0) * v0 * temperatureCorrtedDensity(p_a.temperature);
			real_t density = wm0;
			real_t volume = ker_W(0) * pow(p_a.volume, 2);
			real_t tempdensity = wm0;

			const Neigb* neigbs = f_neigbs[i].neigs; int n = f_neigbs[i].num;
			// forearch neighbour for computing rho
			for (int j = 0; j < n; ++j) {
				if (neigbs[j].pidx.isFluid()) { // fluid neighbour
					const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);
					density += p_b.fm0 * ker_W(neigbs[j].dis);
					tempdensity += p_b.fm0 * ker_W(neigbs[j].dis);
					volume += pow(p_b.volume, 2) * ker_W(neigbs[j].dis);
					gamma += p_b.volume * ker_W(neigbs[j].dis);
				}
				else if (neigbs[j].pidx.isCandidate()) { //see 11
					const CandidatePart& p_c = getCandidatePartOfIdx(neigbs[j].pidx);
					density += p_a.rho0 * p_c.volume * ker_W(neigbs[j].dis);
					tempdensity += p_a.rho0 * p_c.volume * ker_W(neigbs[j].dis);
					volume += p_c.volume * p_c.volume * ker_W(neigbs[j].dis);
					gamma += p_c.volume * ker_W(neigbs[j].dis);
				}
				else { // boundary neighbour
					const BoundPart& p_b = getBoundPartOfIdx(neigbs[j].pidx);
					density += p_a.rho0 * p_b.volume * ker_W(neigbs[j].dis);
					tempdensity += p_a.rho0 * p_b.volume * ker_W(neigbs[j].dis);
					volume += p_b.volume * p_b.volume * ker_W(neigbs[j].dis);
					gamma += p_b.volume * ker_W(neigbs[j].dis);
				}
			}

			p_a.density = density;
			p_a.volume = gamma;
		}

#pragma omp parallel for
		for (int i = 0; i < num; ++i) {
			vec_t tempValue1 = vec_t::O;
			vec_t grad = vec_t::O;
			vec_t ni = vec_t::O;
			FluidPart& p_a = f_parts[i];
			real_t rho0 = p_a.beta * rho0;
			const Neigb* neigbs = f_neigbs[i].neigs; int n = f_neigbs[i].num;
			// forearch neighbour for computing dii
			for (int j = 0; j < n; ++j) {
				grad = vec_t::O;
				if (neigbs[j].pidx.isFluid()) { // fluid neighbour
					const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);
					grad = (p_a.position - p_b.position) * (-ker_W_grad(neigbs[j].dis) / neigbs[j].dis);
					tempValue1 += grad * (-p_b.fm0 / (p_a.density * p_a.density));
					//compute ni(表面张力)
					ni += grad * (p_b.fm0 / p_b.density); // question
				}
				else if (neigbs[j].pidx.isCandidate()) {//see 11
					const CandidatePart& p_c = getCandidatePartOfIdx(neigbs[j].pidx);
					grad = (p_a.position - p_c.position) * (-ker_W_grad(neigbs[j].dis) / neigbs[j].dis);
					tempValue1 += grad * (-(rho0_ori * p_c.volume) / (p_a.density * p_a.density));
				}
				else { // boundary neighbour
					const BoundPart& p_b = getBoundPartOfIdx(neigbs[j].pidx);
					grad = (p_a.position - p_b.position) * (-ker_W_grad(neigbs[j].dis) / neigbs[j].dis);
					tempValue1 += grad * (-(rho0_ori * p_b.volume) / (p_a.density * p_a.density));
				}
			}
			p_a.dii = tempValue1 * m_TH.dt * m_TH.dt;
			ni = ni * m_TH.h;
			p_a.n = ni;

			//compute vel_adv
			p_a.vel_adv = p_a.velocity + p_a.acce_adv * m_TH.dt;
		}
		real_t min_aii=-2;
		real_t aii_dii = 0;
		real_t aii_dji = 0;
		// second loop for foreach particle 
#pragma omp parallel for
		for (int i = 0; i < num; ++i) {
			real_t tempValue1 = 0;
			real_t tempValue2 = 0;
			vec_t grad = vec_t::O;
			vec_t dji = vec_t::O;

			FluidPart& p_a = f_parts[i];
			real_t rho0 = p_a.beta * rho0_ori;
			real_t fm0 = rho0 * p_a.volume;

			const Neigb* neigbs = f_neigbs[i].neigs; int n = f_neigbs[i].num;
			// forearch neighbour for computing rho_adv,aii
			for (int j = 0; j < n; ++j) {
				grad = vec_t::O;
				dji = vec_t::O;
				if (neigbs[j].pidx.isFluid()) { // fluid neighbour
					const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);
					grad = (p_a.position - p_b.position) * (-ker_W_grad(neigbs[j].dis) / neigbs[j].dis);
					tempValue1 += grad.dot((p_a.vel_adv - p_b.vel_adv)) * p_b.volume * m_TH.dt;
					dji = grad * (p_a.fm0 * m_TH.dt * m_TH.dt / (p_a.density * p_a.density));
					tempValue2 += (p_a.dii - dji).dot(grad) * p_b.fm0;

				}
				else if (neigbs[j].pidx.isCandidate()) {//see 11
					const CandidatePart& p_c = getCandidatePartOfIdx(neigbs[j].pidx);
					grad = (p_a.position - p_c.position) * (-ker_W_grad(neigbs[j].dis) / neigbs[j].dis);
					tempValue1 += grad.dot((p_a.vel_adv - p_c.velocity)) * (p_c.volume) * m_TH.dt;
					// dji = grad * (rho0 * p_c.volume * m_TH.dt * m_TH.dt / (p_a.density * p_a.density));
					tempValue2 += (p_a.dii - dji).dot(grad) * (p_c.volume * rho0_ori);
				}
				else { // boundary neighbour
					const BoundPart& p_b = getBoundPartOfIdx(neigbs[j].pidx);
					grad = (p_a.position - p_b.position) * (-ker_W_grad(neigbs[j].dis) / neigbs[j].dis);
					tempValue1 += grad.dot((p_a.vel_adv - p_b.velocity)) * (p_b.volume) * m_TH.dt;
					// dji = grad * (rho0 * p_b.volume * m_TH.dt * m_TH.dt / (p_a.density * p_a.density));
					tempValue2 += (p_a.dii - dji).dot(grad) * (p_b.volume * rho0_ori);
				}
			}
			if (tempValue2 == 0) { tempValue2 = -1e-15; }
			p_a.aii = tempValue2/rho0;
			p_a.vol_adv = p_a.volume + tempValue1;
			p_a.p_l = 0;
#pragma omp critical
			{
				if (p_a.aii > 0) {
					min_aii = p_a.aii;
					aii_dii = 0;
					aii_dji = 0;
					for (int j = 0; j < n; ++j) {
						grad = vec_t::O;
						dji = vec_t::O;
						if (neigbs[j].pidx.isFluid()) { // fluid neighbour
							const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);
							grad = (p_a.position - p_b.position) * (-ker_W_grad(neigbs[j].dis) / neigbs[j].dis);
							tempValue1 += grad.dot((p_a.vel_adv - p_b.vel_adv)) * p_b.volume * m_TH.dt;
							dji = grad * (p_a.fm0 * m_TH.dt * m_TH.dt / (p_a.density * p_a.density));
							aii_dii += (p_a.dii).dot(grad) * p_b.fm0 / rho0;
							aii_dji += (- dji).dot(grad) * p_b.fm0 / rho0;
						}
					}
				}
			}
		}
		//cout << "positive aii: " << min_aii << endl;
		//cout << "dii part: " << aii_dii << " dji part: " << aii_dji << endl;

		//pressure slover
		int l = 0;
		real_t averageVolumeError = 1e10;
		real_t sumVolume = 0;
		long nbParticlesInSummation = 0;
		double eta = 0.001;
		int maxIter = 5000;
		while ((averageVolumeError > eta || l < 3)) {
			if (l > maxIter) break;

			// third loop for foreach particle
#pragma omp parallel for
			for (int i = 0; i < num; ++i) {
				vec_t tempValue1 = vec_t::O;
				vec_t grad = vec_t::O;

				FluidPart& p_a = f_parts[i];
				real_t rho0 = p_a.beta * rho0_ori;
				real_t fm0 = rho0 * p_a.volume;

				const Neigb* neigbs = f_neigbs[i].neigs; int n = f_neigbs[i].num;
				// forearch neighbour 
				for (int j = 0; j < n; ++j) {
					grad = vec_t::O;
					if (neigbs[j].pidx.isFluid()) { // fluid neighbour
						const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);
						grad = (p_a.position - p_b.position) * (-ker_W_grad(neigbs[j].dis) / neigbs[j].dis);
						tempValue1 += grad * (-(p_b.fm0 * p_b.p_l) / (p_b.density * p_b.density));
					}
					else if (neigbs[j].pidx.isCandidate()) {//see 11
						const CandidatePart& p_c = getCandidatePartOfIdx(neigbs[j].pidx);
						// tempValue1 += grad * (-(rho0 * p_c.volume * p_a.p_l) / (p_a.density * p_a.density)); 
					}
					else { // boundary neighbour
						const BoundPart& p_b = getBoundPartOfIdx(neigbs[j].pidx);
						// tempValue1 += grad * (-(rho0 * p_b.volume * p_a.p_l) / (p_a.density * p_a.density)); 
					}
				}
				p_a.sum_dijpj = tempValue1 * (m_TH.dt * m_TH.dt);
			}
			real_t maxPressure = 0;
			real_t maxBeta = 2;
			real_t probeAii = 0;
			// fourth loop for foreach particle
#pragma omp parallel for
			for (int i = 0; i < num; ++i) {
				vec_t grad = vec_t::O;
				vec_t tempValue1 = vec_t::O;
				real_t finalterm = 0;
				real_t t = 0;

				FluidPart& p_a = f_parts[i];
				real_t rho0 = p_a.beta * rho0_ori;
				real_t fm0 = rho0 * p_a.volume;

				const Neigb* neigbs = f_neigbs[i].neigs; int n = f_neigbs[i].num;
				// forearch neighbour 
				for (int j = 0; j < n; ++j) {
					vec_t dji = vec_t::O;
					grad = vec_t::O;
					tempValue1 = vec_t::O;
					if (neigbs[j].pidx.isFluid()) { // fluid neighbour
						const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);
						grad = (p_a.position - p_b.position) * (-ker_W_grad(neigbs[j].dis) / neigbs[j].dis);
						dji = grad * m_TH.dt * m_TH.dt * p_a.fm0 / (p_a.density * p_a.density);
						tempValue1 = (p_a.sum_dijpj - p_b.dii * p_b.p_l) - (p_b.sum_dijpj - dji * p_a.p_l);
						finalterm += (p_b.fm0 * tempValue1.dot(grad));

					}
					else if (neigbs[j].pidx.isCandidate()) {//see 11
						const CandidatePart& p_c = getCandidatePartOfIdx(neigbs[j].pidx);
						grad = (p_a.position - p_c.position) * (-ker_W_grad(neigbs[j].dis) / neigbs[j].dis);
						// dji = grad * m_TH.dt * m_TH.dt * p_a.fm0 / (p_a.density * p_a.density);
						tempValue1 = (p_a.sum_dijpj);
						finalterm += (p_c.volume * rho0_ori * tempValue1.dot(grad));
					}
					else { // boundary neighbour
						const BoundPart& p_b = getBoundPartOfIdx(neigbs[j].pidx);
						grad = (p_a.position - p_b.position) * (-ker_W_grad(neigbs[j].dis) / neigbs[j].dis);
						// dji = grad * m_TH.dt * m_TH.dt * p_a.fm0 / (p_a.density * p_a.density);
						tempValue1 = (p_a.sum_dijpj);
						finalterm += (p_b.volume * rho0_ori * tempValue1.dot(grad));
					}
				}
				finalterm = finalterm / rho0;
				t = 0.5 * p_a.p_l + (0.5 / p_a.aii) * (1 - p_a.vol_adv - finalterm);
				if (t < 0) {
					t = 0;
				}
				else {
					real_t singleVolume = p_a.vol_adv + p_a.aii * p_a.p_l + finalterm;
#pragma omp critical
					{
						//sumDensities += sumDensities;
						if (p_a.vol_adv - 1 > 0) {
							sumVolume += singleVolume;
							++nbParticlesInSummation;
						}
					}
				}
				p_a.presure = t;
				if (maxPressure < t) { maxPressure = t; maxBeta = p_a.beta; probeAii = p_a.aii; }
			}
			//cout << "max pressure: " << maxPressure << " and beta: " << maxBeta << endl;
			//cout << "and it's aii: " << probeAii << endl;
#pragma omp parallel for
			for (int i = 0; i < num; ++i) {
				FluidPart& p_a = f_parts[i];
				p_a.p_l = p_a.presure;
			}

			l = l + 1;
			if (nbParticlesInSummation > 0) {
				averageVolumeError = ((sumVolume / nbParticlesInSummation) - 1);
				nbParticlesInSummation = 0;
				sumVolume = 0;
			}
			else {
				averageVolumeError = 0.0;
			}
			//cout << "averageVolumeError: " << averageVolumeError << endl;
		}
		// cout << "averageVolumeError: " << averageVolumeError << endl;
		//cout << "numIter: " << l << endl << endl;
	}
}


// paritlce-particle interaction, [Mon92], [BT07]
inline void IISph::ii_fluidPartForceExceptPressure_fsame(
	FluidPart& fa, const FluidPart& fb, const real_t& dis,
	const real_t& fm0, const real_t& alpha, const real_t& gamma)
{
	if (dis == 0) { ++m_EC.zeroDis; return; }
	vec_t xab = fa.position - fb.position;
	real_t grad = -ker_W_grad(dis) / dis;
	real_t acce = 0;
	real_t dis2;
	dis2 = dis;
	// viscosity
	real_t pro = (fa.velocity - fb.velocity).dot(xab);
	if (pro < 0) {
		real_t nu = 2 * alpha * m_TH.smoothRadius_h * m_TH.soundSpeed_cs / (fa.density + fb.density);
		real_t pi = -nu * pro / (dis2 * dis2 + real_t(0.01) * m_TH.smoothRadius_h * m_TH.smoothRadius_h);
		acce += -ker_W_grad(dis2) / dis2 * (-fb.fm0 * pi);
	}
	if (acce) { xab *= acce; fa.acce_adv += xab; }
	// surface tension
	if (m_TH.applyCohesion) {
		// surface tension
		real_t h = m_TH.h;
		real_t r = m_TH.r;
		//compute cohesion
		vec_t xab_coh = fa.position - fb.position;
		double pai = 3.14;
		real_t Kij = (fa.rho0 + fb.rho0) / (fa.density + fb.density);
		real_t C = 0;
		if (2 * dis2 > m_TH.h&& dis2 <= m_TH.h) {
			C = (32 / (pai * pow(m_TH.h, 9))) * pow(m_TH.h - dis2, 3) * pow(dis2, 3);
		}
		else if (dis2 > 0 && 2 * dis2 <= m_TH.h) {
			C = (32 / (pai * pow(m_TH.h, 9))) * (2 * pow(m_TH.h - dis2, 3) * pow(dis2, 3) - pow(m_TH.h, 6) / 64);
		}
		else {
			C = 0;
		}

		real_t acce_coh = -r * fb.fm0 * C / dis2;
		xab_coh *= acce_coh;

		//compute curvature
		real_t acce_cur = -m_TH.r;
		vec_t xab_cur = fa.n - fb.n;
		xab_cur *= acce_cur;

		vec_t acce_st = (xab_coh) * Kij;
		fa.acce_adv += acce_st;
	}
}

// multiphase fluid, [SP08]
inline void IISph::ii_fluidPartForceExceptPressure_fdiff(
	FluidPart& fa, const FluidPart& fb, const real_t& dis,
	const real_t& fma, const real_t& fmb, const real_t& alpha)
{
	if (dis == 0) { ++m_EC.zeroDis; return; }
	vec_t xab = fa.position - fb.position;
	real_t grad = -ker_W_grad(dis) / dis;
	real_t acce = 0;
	// viscosity
	real_t pro = (fa.velocity - fb.velocity).dot(xab);
	if (pro < 0) {
		real_t nu = 2 * alpha * m_TH.smoothRadius_h * m_TH.soundSpeed_cs / (fa.density + fb.density);
		real_t pi = -nu * pro / (dis * dis + real_t(0.01) * m_TH.smoothRadius_h * m_TH.smoothRadius_h);
		acce += grad * (-(fma + fmb) / 2 * pi);
	}
	if (acce) { xab *= acce; fa.acce_adv += xab; }

}

// fluid-rigid coupling, [AIS*12]
inline void IISph::ii_fluidPartForceExceptPressure_bound(
	FluidPart& fa, const BoundPart& rb, const real_t& dis,
	const real_t& frho0, const real_t& r_alpha)
{
	if (dis == 0) { ++m_EC.zeroDis; return; }
	vec_t xab = fa.position - rb.position;
	real_t grad = -ker_W_grad(dis) / dis;
	real_t acce = 0;
	// viscosity
	real_t pro = (fa.velocity - rb.velocity).dot(xab);
	if (pro < 0) {
		real_t nu = 2 * r_alpha * m_TH.smoothRadius_h * m_TH.soundSpeed_cs / (fa.density * 2);
		real_t pi = -nu * pro / (dis * dis + real_t(0.01) * m_TH.smoothRadius_h * m_TH.smoothRadius_h);
		acce += grad * (-frho0 * rb.volume * pi);
	}
	if (acce) { xab *= acce; fa.acce_adv += xab; }
	if (m_TH.applyAdhesion) {
		// surface tension & adhesion
		real_t bt = 1;
		real_t h = m_TH.h;
		real_t A = 0;
		vec_t xab_adh = fa.position - rb.position;

		if (2 * dis > m_TH.h&& dis <= m_TH.h) {
			A = 0.007 / (pow(m_TH.h, 13 / 4)) * pow((-4 * dis * dis / m_TH.h + 6 * dis - 2 * m_TH.h), 1 / 4);
		}
		else {
			A = 0;
		}
		real_t acce_adh = -m_TH.bt * fa.rho0 * rb.volume * A / dis;
		xab_adh *= acce_adh;
		fa.acce_adv += xab_adh;
	}
}

//see 11


void IISph::ii_forceExceptPressure() {
	// foreach fluid
	for (int n_f = int(m_Fluids.size()), k = 0; k < n_f; ++k) {
		std::vector<FluidPart>& f_parts = m_Fluids[k].fluidParticles;
		const std::vector<NeigbStr>& f_neigbs = mg_NeigbOfFluids[k];
		real_t rho0_ori = m_Fluids[k].restDensity_rho0;
		//real_t fm0 = rho0 * v0;
		real_t alpha = m_Fluids[k].viscosity_alpha;
		real_t gamma = m_Fluids[k].surfaceTension_gamma;
		int num = int(f_parts.size());
		// foreach particle
#pragma omp parallel for
		for (int i = 0; i < num; ++i) {
			FluidPart& p_a = f_parts[i];
			p_a.volume = pow(p_a.d, vec_t::dim);
			real_t rho0 = p_a.beta * rho0_ori;
			real_t fm0 = rho0 * p_a.volume;
			p_a.presure = 0;
			p_a.acce_adv = m_TH.gravity_g;
			//p_a.acce_adv = vec_t::O;
			const Neigb* neigbs = f_neigbs[i].neigs; int n = f_neigbs[i].num;
			// forearch neighbour
			for (int j = 0; j < n; ++j) {
				if (neigbs[j].pidx.isFluid()) { // fluid neighbour
					const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);
					int idx_b = neigbs[j].pidx.toFluidI();
					if (idx_b == k) {
						// the same fluid
						ii_fluidPartForceExceptPressure_fsame(
							p_a, p_b, neigbs[j].dis, fm0, alpha, gamma);
					}
					else {
						// different fluid
						//注：这里的p_a.volums是错误的，之后用到different fluid再改
						real_t fmb = p_a.volume * m_Fluids[idx_b].restDensity_rho0;
						real_t b_alpha = m_Fluids[idx_b].viscosity_alpha;
						ii_fluidPartForceExceptPressure_fdiff(
							p_a, p_b, neigbs[j].dis, fm0, fmb, (alpha + b_alpha) / 2);

					}
				}
				else if (neigbs[j].pidx.isCandidate()) {//see 11
					const CandidatePart& p_c = getCandidatePartOfIdx(neigbs[j].pidx);
					int idx_c = neigbs[j].pidx.toCandidateI();
					real_t r_alpha = m_Candidates[idx_c].viscosity_alpha;
					computeForceFromCandidateToFluidExceptPressure(p_a, p_c, neigbs[j].dis, rho0, r_alpha);
				}
				else { // boundary neighbour
					const BoundPart& p_b = getBoundPartOfIdx(neigbs[j].pidx);
					int idx_b = neigbs[j].pidx.toSolidI();
					real_t r_alpha = m_Solids[idx_b].viscosity_alpha;
					ii_fluidPartForceExceptPressure_bound(
						p_a, p_b, neigbs[j].dis, rho0, r_alpha);
				}
			}
		}
	}
}

// paritlce-particle interaction, [Mon92], [BT07]
inline void IISph::ii_fluidPartForcePressure_fsame(
	FluidPart& fa, const FluidPart& fb, const real_t& dis,
	const real_t& fm0, const real_t& alpha, const real_t& gamma)
{
	real_t dis2;
	dis2 = dis;
	if (dis == 0) { ++m_EC.zeroDis; return; }
	vec_t xab = fa.position - fb.position;
	real_t grad = -ker_W_grad(dis) / dis;
	// momentum
	real_t acce = grad
		* (-fb.fm0 * (fa.presure / (fa.density * fa.density) + fb.presure / (fb.density * fb.density)));
	xab *= acce; fa.acce_presure += xab;
}

// multiphase fluid, [SP08]
inline void IISph::ii_fluidPartForcePressure_fdiff(
	FluidPart& fa, const FluidPart& fb, const real_t& dis,
	const real_t& fma, const real_t& fmb, const real_t& alpha)
{
	if (dis == 0) { ++m_EC.zeroDis; return; }
	vec_t xab = fa.position - fb.position;
	real_t grad = -ker_W_grad(dis) / dis;
	real_t dalta_a = fa.density / fma, dalta_b = fb.density / fmb;
	// momentum
	real_t acce = grad
		* (-1 / fma * (fa.presure / (dalta_a * dalta_a) + fb.presure / (dalta_b * dalta_b)));
	xab *= acce; fa.acce_presure += xab;
}

// fluid-rigid coupling, [AIS*12]
inline void IISph::ii_fluidPartForcePressure_bound(
	FluidPart& fa, const BoundPart& rb, const real_t& dis,
	const real_t& frho0, const real_t& r_alpha)
{
	if (dis == 0) { ++m_EC.zeroDis; return; }
	vec_t xab = fa.position - rb.position;
	real_t grad = -ker_W_grad(dis) / dis, acce = 0;
	// momentum
	if (fa.presure > 0) {
		acce += grad * (-frho0 * rb.volume * (fa.presure / (fa.density * fa.density) * 2));
	}
	xab *= acce; fa.acce_presure += xab;
}

//see 12
real_t IISph::temperatureCorrtedDensity(real_t temperature) {
	real_t rate = 1 + (temperature - 273.15) / 100;
	return rate;
}

//see 11
void IISph::computeForceFromCandidateToFluidExceptPressure(FluidPart& fa, const CandidatePart& cb, const real_t& dis,
	const real_t& frho0, const real_t& r_alpha) {
	if (dis == 0) { ++m_EC.zeroDis; return; }
	vec_t xab = fa.position - cb.position;
	real_t grad = -ker_W_grad(dis) / dis;
	real_t acce = 0;
	// viscosity
	real_t pro = (fa.velocity - cb.velocity).dot(xab);
	if (pro < 0) {
		real_t nu = 2 * r_alpha * m_TH.smoothRadius_h * m_TH.soundSpeed_cs / (fa.density * 2);
		real_t pi = -nu * pro / (dis * dis + real_t(0.01) * m_TH.smoothRadius_h * m_TH.smoothRadius_h);
		acce += grad * (-frho0 * cb.volume * pi);
	}
	if (acce) { xab *= acce; fa.acce_adv += xab; }
	// surface tension & adhesion
	if (m_TH.applyAdhesion) {
		real_t bt = 1;
		real_t h = m_TH.h;
		real_t A = 0;
		vec_t xab_adh = fa.position - cb.position;

		if (2 * dis > m_TH.h && dis <= m_TH.h) {
			A = 0.007 / (pow(m_TH.h, 13 / 4)) * pow((-4 * dis * dis / m_TH.h + 6 * dis - 2 * m_TH.h), 1 / 4);
		}
		else {
			A = 0;
		}
		real_t acce_adh = -m_TH.bt * frho0 * cb.volume * A / dis;
		xab_adh *= acce_adh;
		fa.acce_presure += xab_adh;
	}
}

//see 11
void IISph::computeForceFromCandidateToFluid(FluidPart& fa, const CandidatePart& cb, const real_t& dis,
	const real_t& frho0, const real_t& r_alpha) {
	if (dis == 0) { ++m_EC.zeroDis; return; }
	vec_t xab = fa.position - cb.position;
	real_t grad = -ker_W_grad(dis) / dis, acce = 0;
	// momentum
	if (fa.presure > 0)
		acce += grad * (-frho0 * cb.volume * (fa.presure / (fa.density * fa.density) * 2));
	xab *= acce; fa.acce_presure += xab;
}

//see 11
void IISph::computeForceFromFluidToCandidate(CandidatePart& ca, const FluidPart& fb, const real_t& dis,
	const real_t& frho0, const real_t& fm0, const real_t& r_alpha) {
	if (dis == 0) { ++m_EC.zeroDis; return; }

	vec_t xab = ca.position - fb.position;
	real_t grad = -ker_W_grad(dis) / dis, force = 0;
	// momentum
	if (fb.presure > 0)
		force += grad * (-fm0 * frho0 * ca.volume * (fb.presure / (fb.density * fb.density) * 2));
	// viscosity

	real_t pro = (ca.velocity - fb.velocity).dot(xab);
	if (pro < 0) {
		real_t nu = 2 * r_alpha * m_TH.smoothRadius_h * m_TH.soundSpeed_cs / (fb.density * 2);
		real_t pi =
			-nu * pro / (dis * dis + real_t(0.01) * m_TH.smoothRadius_h * m_TH.smoothRadius_h);
		force += grad * (-fm0 * frho0 * ca.volume * pi);
	}
	xab *= force; ca.force += xab;

	if (m_TH.applyAdhesion) {
		// surface tension & adhesion
		real_t bt = 1;
		real_t h = m_TH.h;
		real_t A = 0;
		vec_t xab_adh = ca.position - fb.position;

		if (2 * dis > m_TH.h&& dis <= m_TH.h) {
			A = 0.007 / (pow(m_TH.h, 13 / 4)) * pow((-4 * dis * dis / m_TH.h + 6 * dis - 2 * m_TH.h), 1 / 4);
		}
		else {
			A = 0;
		}
		real_t acce_adh = -m_TH.bt * frho0 * ca.volume * A / dis;
		xab_adh *= acce_adh;
		ca.force += (xab_adh * ca.mass);
	}
}

//see 11
void IISph::computeForceFromSolidToCandidate(CandidatePart& ca, const BoundPart& bb, const real_t& dis, const real_t& r_alpha) {
	real_t grad = -ker_W_grad(dis) / dis;
	vec_t xab = ca.position - bb.position;
	vec_t vab = ca.velocity - bb.velocity;
	real_t mass = ca.mass;
	vec_t force = vec_t(0, 0, 0);
	if (xab.dot(vab) < 0)
		force = xab.normalize() * pow(m_TH.spacing_r / (dis - 0.5 * m_TH.spacing_r), 6) * 600;
	else {
		force = xab.normalize() * pow(m_TH.spacing_r / (dis - 0.5 * m_TH.spacing_r), 6) * 100;
	}
	real_t force2;
	ca.force += force;
}
//see 11
void IISph::computeForceFromCandidateToCandidate(CandidatePart& cp1, CandidatePart& cp2, const real_t& dis) {
	vec_t xab = cp2.position - cp1.position;
	vec_t force = vec_t(0, 0, 0);
	vec_t vab = cp2.velocity - cp1.velocity;
	if (xab.dot(vab) < 0)
		force = xab.normalize() * pow(m_TH.spacing_r / (dis - 0.5 * m_TH.spacing_r), 6) * 600;
	else {
		force = xab.normalize() * pow(m_TH.spacing_r / (dis - 0.5 * m_TH.spacing_r), 6) * 100;
	}
	real_t force2;
	cp2.force += force;
}


// fluid-rigid coupling, [AIS*12]
inline void IISph::ii_boundPartForcePressure_f(
	BoundPart& ra, const FluidPart& fb, const real_t& dis,
	const real_t& frho0, const real_t& fm0, const real_t& r_alpha)
{
	if (dis == 0) { ++m_EC.zeroDis; return; }

	vec_t xab = ra.position - fb.position;
	real_t grad = -ker_W_grad(dis) / dis, force = 0;
	// momentum
	if (fb.presure > 0)
		force += grad * (-fm0 * frho0 * ra.volume * (fb.presure / (fb.density * fb.density) * 2));
	// viscosity
	real_t pro = (ra.velocity - fb.velocity).dot(xab);
	if (pro < 0) {
		real_t nu = 2 * r_alpha * m_TH.smoothRadius_h * m_TH.soundSpeed_cs / (fb.density * 2);
		real_t pi =
			-nu * pro / (dis * dis + real_t(0.01) * m_TH.smoothRadius_h * m_TH.smoothRadius_h);
		force += grad * (-fm0 * frho0 * ra.volume * pi);
	}
	xab *= force; ra.force += xab;

}
// compute gravity, pressure and friction force
void IISph::ii_computeForcePressure()
{
	real_t v0 = std::pow(m_TH.spacing_r, vec_t::dim);
	// foreach fluid
	for (int n_f = int(m_Fluids.size()), k = 0; k < n_f; ++k) {
		std::vector<FluidPart>& f_parts = m_Fluids[k].fluidParticles;
		const std::vector<NeigbStr>& f_neigbs = mg_NeigbOfFluids[k];
		real_t rho0_ori = m_Fluids[k].restDensity_rho0;
		//real_t fm0 = rho0 * v0;
		real_t alpha = m_Fluids[k].viscosity_alpha;
		real_t gamma = m_Fluids[k].surfaceTension_gamma;
		int num = int(f_parts.size());
		// foreach particle
#pragma omp parallel for 
		for (int i = 0; i < num; ++i) {
			FluidPart& p_a = f_parts[i];

			real_t rho0 = p_a.beta * rho0_ori;
			real_t fm0 = rho0 * v0;
			p_a.acce_presure = p_a.acce_adv;
			
			const Neigb* neigbs = f_neigbs[i].neigs; int n = f_neigbs[i].num;
			// forearch neighbour
			for (int j = 0; j < n; ++j) {
				if (neigbs[j].pidx.isFluid()) { // fluid neighbour
					const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);
					int idx_b = neigbs[j].pidx.toFluidI();
					// the same fluid
					if (idx_b == k) {
						ii_fluidPartForcePressure_fsame(p_a, p_b, neigbs[j].dis, fm0, alpha, gamma);
						// different fluid
					}
					else {
						real_t fmb = v0 * m_Fluids[idx_b].restDensity_rho0;
						real_t b_alpha = m_Fluids[idx_b].viscosity_alpha;
						ii_fluidPartForcePressure_fdiff(
							p_a, p_b, neigbs[j].dis, fm0, fmb, (alpha + b_alpha) / 2);
					}
				}
				else if (neigbs[j].pidx.isCandidate()) {//see 11
					const CandidatePart& p_c = getCandidatePartOfIdx(neigbs[j].pidx);
					int idx_c = neigbs[j].pidx.toCandidateI();
					real_t r_alpha = m_Candidates[idx_c].viscosity_alpha;
					computeForceFromCandidateToFluid(p_a, p_c, neigbs[j].dis, rho0, 0.5);
				}
				else { // boundary neighbour
					const BoundPart& p_b = getBoundPartOfIdx(neigbs[j].pidx);
					int idx_b = neigbs[j].pidx.toSolidI();
					real_t r_alpha = m_Solids[idx_b].viscosity_alpha;
					ii_fluidPartForcePressure_bound(p_a, p_b, neigbs[j].dis, rho0, r_alpha);
				}
			}//nneighbour
#ifdef II_ADT
			 // the h in the paper is r, and the smoothing_h is 2h
			real_t v = !(p_a.velocity == vec_t::O)
				? m_Lambda_v * (m_TH.spacing_r / p_a.velocity.length()) : 1;
			real_t f = !(p_a.acceleration == vec_t::O)
				? m_Lambda_f * std::sqrt(m_TH.spacing_r / p_a.acceleration.length()) : 1;
			p_a.dt = std::min(v, f);
#endif
		}//particle
	}//fluid

	 // forearch rigid
	for (int n_r = int(m_Solids.size()), k = 0; k < n_r; ++k) if (m_Solids[k].dynamic) {
		std::vector<BoundPart>& r_parts = m_Solids[k].boundaryParticles;
		const std::vector<NeigbStr>& r_neigbs = mg_NeigbOfSolids[k];
		real_t alpha = m_Solids[k].viscosity_alpha;
		int num = int(r_parts.size());
		// foreach particle
#pragma omp parallel for
		for (int i = 0; i < num; ++i) {
			BoundPart& p_a = r_parts[i];
			p_a.force = vec_t::O;
			const Neigb* neigbs = r_neigbs[i].neigs; int n = r_neigbs[i].num;
			// forearch neighbour
			for (int j = 0; j < n; ++j) {
				if (neigbs[j].pidx.isFluid()) { // fluid neighbour
					const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);
					int idx_b = neigbs[j].pidx.toFluidI();
					real_t frho0 = m_Fluids[idx_b].restDensity_rho0 * p_b.beta;
					ii_boundPartForcePressure_f(p_a, p_b, neigbs[j].dis, frho0, frho0 * v0, alpha);
				}
			}//neighbour
		}//particle
	}//rigid

	 //see 11 for each Candidate
	for (int n_c = int(m_Candidates.size()), k = 0; k < n_c; k++) {
		if (m_Candidates[k].dynamic) {
			std::vector<CandidatePart>& c_parts = m_Candidates[k].candidateParticles;
			const std::vector<NeigbStr>& c_neigbs = mg_NeigbOfCandidates[k];
			real_t alpha = m_Candidates[k].viscosity_alpha;
			int num = int(c_parts.size());
#pragma omp parallel for
			//并行之前保留
			for (int i = 0; i < num; ++i) {
				CandidatePart& p_c = c_parts[i];
				p_c.force = vec_t(0, 0, 0);
				const Neigb* neigbs = c_neigbs[i].neigs;
				int n = c_neigbs[i].num;
				for (int j = 0; j < n; ++j) {
					if (neigbs[j].pidx.isFluid()) { // fluid neighbour
						const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);
						int idx_b = neigbs[j].pidx.toFluidI();
						real_t frho0 = m_Fluids[idx_b].restDensity_rho0 * p_b.beta;
						computeForceFromFluidToCandidate(p_c, p_b, neigbs[j].dis, frho0, frho0 * v0, 0.5);
					}
					else if (neigbs[j].pidx.isSolid()) {
						const BoundPart& p_a = getBoundPartOfIdx(neigbs[j].pidx);
						int idx_b = neigbs[j].pidx.toSolidI();
						computeForceFromSolidToCandidate(p_c, p_a, neigbs[j].dis, alpha);
					}
					else if (neigbs[j].pidx.isCandidate()) {
						if (k != neigbs[j].pidx.toCandidateI()) {
							CandidatePart& pc2 = m_Candidates[neigbs[j].pidx.toCandidateI()].candidateParticles[neigbs[j].pidx.i];
							int idx_c = neigbs[j].pidx.toCandidateI();
							real_t r_alpha = m_Candidates[idx_c].viscosity_alpha;
							computeForceFromCandidateToCandidate(pc2, p_c, neigbs[j].dis);
						}
					}
				}
			}
		}
	}

}

inline void IISph::fluidToSolid(int fluidIdx, const char* meshFileName,
	const char* sampleFileName, real_t viscosity, const glm::mat4& transform) {

}