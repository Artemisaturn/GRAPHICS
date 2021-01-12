
#define _SILENCE_STDEXT_HASH_DEPRECATION_WARNINGS 1

#include "IISph.h"
#include <iomanip>
#include "iostream"
#include <fstream>
#include <cmath>

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

	CALL_TIME(DF_prepareAttribute());
	CALL_TIME(VF_prepareAttribute());

	CALL_TIME(advectionStep());

	//CALL_TIME(DF_divergenceFreeSolver());
	//CALL_TIME(DF_constantDensitySolver());


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

	for (int n_f = int(m_Fluids.size()), k = 0; k < n_f; ++k) {

		std::vector<FluidPart>& f_parts = m_Fluids[k].fluidParticles;
		const std::vector<NeigbStr>& f_neigbs = mg_NeigbOfFluids[k];

		real_t rho0 = m_Fluids[k].restDensity_rho0;
		int num = int(f_parts.size());

#pragma omp parallel for
		for (int i = 0; i < num; ++i) {
			FluidPart& p_a = f_parts[i];
			p_a.restVolume = std::pow(p_a.d, vec_t::dim);
			p_a.restDensity = rho0 * p_a.beta;
			p_a.mass = p_a.restDensity * p_a.restVolume;
			p_a.advectionAcceleration = vec_t(0, 0, 0);
		}
	}

	for (int n_b = int(m_Solids.size()), k = 0; k < n_b; ++k) {

		std::vector<BoundPart>& b_parts = m_Solids[k].boundaryParticles;
		const std::vector<NeigbStr>& b_neigbs = mg_NeigbOfSolids[k];

		int num = int(b_parts.size());

#pragma omp parallel for
		for (int i = 0; i < num; ++i) {
			BoundPart& p_a = b_parts[i];
			p_a.mass = p_a.restDensity * p_a.volume;
			p_a.advectionAcceleration = vec_t(0, 0, 0);
		}
	}

	for (int n_f = int(m_Fluids.size()), k = 0; k < n_f; ++k) {

		std::vector<FluidPart>& f_parts = m_Fluids[k].fluidParticles;
		const std::vector<NeigbStr>& f_neigbs = mg_NeigbOfFluids[k];

		real_t rho0 = m_Fluids[k].restDensity_rho0;
		int num = int(f_parts.size());

#pragma omp parallel for
		for (int i = 0; i < num; ++i) {
			FluidPart& p_a = f_parts[i];

			vec_t grad = vec_t::O;
			real_t gradScalar = 0;
			real_t h = p_a.d;
			real_t density = ker_W(0, h) * p_a.mass;

			p_a.DF_alpha = 0;
			p_a.DF_alpha1 = vec_t(0, 0, 0);
			p_a.DF_alpha2 = 0;

			const Neigb* neigbs = f_neigbs[i].neigs; int n = f_neigbs[i].num;

			for (int j = 0; j < n; ++j) {
				if (neigbs[j].pidx.isFluid()) {
					const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);

					h = (p_a.d + p_b.d) / 2;
					if (m_TH.adjustDensity) { density += p_a.mass * ker_W(neigbs[j].dis, h); }
					else{ density += p_b.mass * ker_W(neigbs[j].dis, h); }

					gradScalar = -ker_W_grad(neigbs[j].dis, h);
					grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

					p_a.DF_alpha1 += grad * p_b.mass;
					p_a.DF_alpha2 += pow(gradScalar * p_b.mass, 2);
				}
				else if (neigbs[j].pidx.isCandidate()) {
					const CandidatePart& p_c = getCandidatePartOfIdx(neigbs[j].pidx);

					h = (p_a.d + p_c.d) / 2;
					density += p_a.restDensity * p_c.volume * ker_W(neigbs[j].dis, h);

					gradScalar = -ker_W_grad(neigbs[j].dis, h);
					grad = (p_a.position - p_c.position) * (gradScalar / neigbs[j].dis);

					p_a.DF_alpha1 += grad * p_a.restDensity * p_c.volume;
				}
				else{
					const BoundPart& p_b = getBoundPartOfIdx(neigbs[j].pidx);

					h = (p_a.d + p_b.d) / 2;
					density += p_a.restDensity * p_b.volume * ker_W(neigbs[j].dis, h);

					gradScalar = -ker_W_grad(neigbs[j].dis, h);
					grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

					p_a.DF_alpha1 += grad * m_TH.boundaryDensity * p_b.volume;
					if (m_Solids[neigbs[j].pidx.toSolidI()].dynamic) {
						p_a.DF_alpha2 += pow(gradScalar * p_b.mass, 2);
					}
				}
			}
			p_a.sphDensity = density;
			p_a.DF_alpha = p_a.DF_alpha1.len_square() + p_a.DF_alpha2;
			if (p_a.DF_alpha < 1.0e-6) {
				p_a.DF_alpha = 1.0e-6;
			}
			p_a.DF_alpha = p_a.sphDensity / p_a.DF_alpha;
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
					real_t h = (p_a.d + p_b.d) / 2;
					grad = (p_a.position - p_b.position) * (-ker_W_grad(neigbs[j].dis, h) / neigbs[j].dis);
					ni += grad * (p_b.mass / p_b.sphDensity); // question
				}
			}
			p_a.n = ni * m_TH.h;
		}
	}

	for (int n_b = int(m_Solids.size()), k = 0; k < n_b; ++k) {

		std::vector<BoundPart>& b_parts = m_Solids[k].boundaryParticles;
		const std::vector<NeigbStr>& b_neigbs = mg_NeigbOfSolids[k];

		int num = int(b_parts.size());

#pragma omp parallel for
		for (int i = 0; i < num; ++i) {
			BoundPart& p_a = b_parts[i];

			vec_t grad = vec_t::O;
			real_t gradScalar = 0;
			real_t h = p_a.d;
			real_t density = ker_W(0, h) * p_a.mass;

			p_a.DF_alpha = 0;
			p_a.DF_alpha1 = vec_t(0, 0, 0);
			p_a.DF_alpha2 = 0;

			const Neigb* neigbs = b_neigbs[i].neigs; int n = b_neigbs[i].num;

			for (int j = 0; j < n; ++j) {
				if (neigbs[j].pidx.isFluid()) {
					const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);

					h = (p_a.d + p_b.d) / 2;
					if (m_TH.adjustDensity) { density += p_a.mass * ker_W(neigbs[j].dis, h); }
					else { density += p_b.mass * ker_W(neigbs[j].dis, h); }

					gradScalar = -ker_W_grad(neigbs[j].dis, h);
					grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

					p_a.DF_alpha2 += pow(gradScalar * p_b.mass, 2);
				}
				else if (neigbs[j].pidx.isCandidate()) {
					const CandidatePart& p_c = getCandidatePartOfIdx(neigbs[j].pidx);

					h = (p_a.d + p_c.d) / 2; 
					density += p_a.restDensity * p_c.volume * ker_W(neigbs[j].dis, h);

					gradScalar = -ker_W_grad(neigbs[j].dis, h);
					grad = (p_a.position - p_c.position) * (gradScalar / neigbs[j].dis);

				}
				else {
					const BoundPart& p_b = getBoundPartOfIdx(neigbs[j].pidx);

					h = (p_a.d + p_b.d) / 2;
					density += p_a.restDensity * p_b.volume * ker_W(neigbs[j].dis, h);

					gradScalar = -ker_W_grad(neigbs[j].dis, h);
					grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);
				}
			}
			p_a.sphDensity = density;
			p_a.DF_alpha = p_a.DF_alpha2;
			if (p_a.DF_alpha < 1.0e-6) {
				p_a.DF_alpha = 1.0e-6;
			}
			p_a.DF_alpha = p_a.sphDensity / p_a.DF_alpha;
		}
	}
}

void IISph::VF_prepareAttribute() {

	for (int n_f = int(m_Fluids.size()), k = 0; k < n_f; ++k) {

		std::vector<FluidPart>& f_parts = m_Fluids[k].fluidParticles;
		const std::vector<NeigbStr>& f_neigbs = mg_NeigbOfFluids[k];

		real_t rho0 = m_Fluids[k].restDensity_rho0;
		int num = int(f_parts.size());

#pragma omp parallel for
		for (int i = 0; i < num; ++i) {
			FluidPart& p_a = f_parts[i];
			p_a.restVolume = std::pow(p_a.d, vec_t::dim);
			p_a.restDensity = rho0 * p_a.beta;
			p_a.mass = p_a.restDensity * p_a.restVolume;
			p_a.advectionAcceleration = vec_t(0, 0, 0);
		}
	}

	for (int n_f = int(m_Fluids.size()), k = 0; k < n_f; ++k) {

		std::vector<FluidPart>& f_parts = m_Fluids[k].fluidParticles;
		const std::vector<NeigbStr>& f_neigbs = mg_NeigbOfFluids[k];

		real_t rho0 = m_Fluids[k].restDensity_rho0;
		int num = int(f_parts.size());
#pragma omp parallel for
		for (int i = 0; i < num; ++i) {
			FluidPart& p_a = f_parts[i];

			vec_t grad = vec_t::O;
			real_t gradScalar = 0;
			real_t h = p_a.d;
			real_t VF_sphVolume = ker_W(0, h) * pow(p_a.restVolume, 2);
			real_t gamma = p_a.restVolume * ker_W(0, h);

			p_a.VF_sphVolume = 0;
			p_a.VF_alpha = 0;
			p_a.VF_alpha1 = vec_t(0, 0, 0);
			p_a.VF_gamma = 0;

			const Neigb* neigbs = f_neigbs[i].neigs; int n = f_neigbs[i].num;

			for (int j = 0; j < n; ++j) {

				if (neigbs[j].pidx.isFluid()) {
					const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);

					h = (p_a.d + p_b.d) / 2;
					gamma += p_b.restVolume * ker_W(neigbs[j].dis, h);
					/*if (i == 1) {
						cout << "ker_W(neigbs[j].dis)_" << j << " : "<< ker_W(neigbs[j].dis, h) << endl;
					}*/
					VF_sphVolume += p_a.restVolume * p_b.restVolume * ker_W(neigbs[j].dis, h);

					gradScalar = -ker_W_grad(neigbs[j].dis, h);
					grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

					p_a.VF_alpha1 += grad * p_b.restVolume;
				}
				else if (neigbs[j].pidx.isCandidate()) {
					const CandidatePart& p_c = getCandidatePartOfIdx(neigbs[j].pidx);

					h = (p_a.d + p_c.d) / 2;
					gamma += p_c.volume * ker_W(neigbs[j].dis, h);
					VF_sphVolume += p_a.restVolume * p_c.volume * ker_W(neigbs[j].dis, h);

					gradScalar = -ker_W_grad(neigbs[j].dis, h);
					grad = (p_a.position - p_c.position) * (gradScalar / neigbs[j].dis);

					p_a.VF_alpha1 += grad * p_c.volume;
				}
				else {
					const BoundPart& p_b = getBoundPartOfIdx(neigbs[j].pidx);

					h = (p_a.d + p_b.d) / 2;
					gamma += p_b.volume * ker_W(neigbs[j].dis, h);
					VF_sphVolume += p_a.restVolume * p_b.volume * ker_W(neigbs[j].dis, h);

					gradScalar = -ker_W_grad(neigbs[j].dis, h);
					grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

					p_a.VF_alpha1 += grad * p_b.volume;
				}
			}

			/*if (i < 10) {
				cout << "gamma_" << i << ": " << gamma << endl;
			}*/
			p_a.VF_gamma = gamma;
			p_a.VF_sphVolume = VF_sphVolume;
		}

#pragma omp parallel for
		for (int i = 0; i < num; ++i) {
			FluidPart& p_a = f_parts[i];

			vec_t grad = vec_t::O;
			real_t gradScalar = 0;
			real_t h = p_a.d;

			p_a.VF_alpha2 = 0;

			const Neigb* neigbs = f_neigbs[i].neigs; int n = f_neigbs[i].num;

			for (int j = 0; j < n; ++j) {

				if (neigbs[j].pidx.isFluid()) {
					const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);
					h = (p_a.d + p_b.d) / 2;
					gradScalar = -ker_W_grad(neigbs[j].dis, h);
					grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

					p_a.VF_alpha2 += pow(p_b.restVolume, 2) * grad.dot(grad) / p_b.mass;
				}
				else if (neigbs[j].pidx.isCandidate()) {
					const CandidatePart& p_c = getCandidatePartOfIdx(neigbs[j].pidx);
					h = (p_a.d + p_c.d) / 2;
					gradScalar = -ker_W_grad(neigbs[j].dis, h);
					grad = (p_a.position - p_c.position) * (gradScalar / neigbs[j].dis);

					p_a.VF_alpha2 += pow(p_c.volume, 2) * grad.dot(grad) / (p_a.restDensity * p_c.volume);
				}
				else {
					const BoundPart& p_b = getBoundPartOfIdx(neigbs[j].pidx);
					h = (p_a.d + p_b.d) / 2;
					gradScalar = -ker_W_grad(neigbs[j].dis, h);
					grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

					if (m_Solids[neigbs[j].pidx.toSolidI()].dynamic) {
						p_a.VF_alpha2 += pow(p_b.volume, 2) * grad.dot(grad) / (p_b.restDensity * p_b.volume);
					}
				}
			}

			p_a.VF_alpha = ((p_a.VF_alpha1.dot(p_a.VF_alpha1) / p_a.mass) + p_a.VF_alpha2) / p_a.VF_gamma;
			if (p_a.VF_alpha < 1.0e-10) {
				p_a.VF_alpha = 1.0e-10;
			}
			p_a.VF_alpha = 1 / p_a.VF_alpha;
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
					real_t h = (p_a.d + p_b.d) / 2;
					grad = (p_a.position - p_b.position) * (-ker_W_grad(neigbs[j].dis, h) / neigbs[j].dis);
					ni += grad * (1 / p_b.VF_sphVolume); // question
				}
			}
			p_a.n = ni * m_TH.h;
		}
	}

	for (int n_b = int(m_Solids.size()), k = 0; k < n_b; ++k) {

		std::vector<BoundPart>& b_parts = m_Solids[k].boundaryParticles;
		const std::vector<NeigbStr>& b_neigbs = mg_NeigbOfSolids[k];
		int num = int(b_parts.size());

#pragma omp parallel for
		for (int i = 0; i < num; ++i) {
			BoundPart& p_a = b_parts[i];

			vec_t grad = vec_t::O;
			real_t gradScalar = 0;
			real_t h = p_a.d;
			real_t VF_sphVolume = ker_W(0, h) * pow(p_a.volume, 2);
			real_t gamma = p_a.volume * ker_W(0, h);

			p_a.VF_sphVolume = 0;
			p_a.VF_gamma = 0;
			p_a.force = vec_t::O;

			const Neigb* neigbs = b_neigbs[i].neigs; int n = b_neigbs[i].num;

			for (int j = 0; j < n; ++j) {

				if (neigbs[j].pidx.isFluid()) {
					const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);
					h = (p_a.d + p_b.d) / 2;
					gamma += p_b.restVolume * ker_W(neigbs[j].dis, h);
					VF_sphVolume += p_a.volume * p_b.restVolume * ker_W(neigbs[j].dis, h);
				}
				else if (neigbs[j].pidx.isCandidate()) {
					const CandidatePart& p_c = getCandidatePartOfIdx(neigbs[j].pidx);
					h = (p_a.d + p_c.d) / 2;
					gamma += p_c.volume * ker_W(neigbs[j].dis, h);
					VF_sphVolume += p_a.volume * p_c.volume * ker_W(neigbs[j].dis, h);
				}
				else {
					const BoundPart& p_b = getBoundPartOfIdx(neigbs[j].pidx);
					h = (p_a.d + p_b.d) / 2;
					gamma += p_b.volume * ker_W(neigbs[j].dis, h);
					VF_sphVolume += p_a.volume * p_b.volume * ker_W(neigbs[j].dis, h);
				}
			}
			p_a.VF_gamma = gamma;
			p_a.VF_sphVolume = VF_sphVolume;
		}

#pragma omp parallel for
		for (int i = 0; i < num; ++i) {
			BoundPart& p_a = b_parts[i];

			vec_t grad = vec_t::O;
			real_t gradScalar = 0;
			real_t h = p_a.d;

			p_a.VF_alpha1 = vec_t(0, 0, 0);
			p_a.VF_alpha2 = 0;

			const Neigb* neigbs = b_neigbs[i].neigs; int n = b_neigbs[i].num;

			for (int j = 0; j < n; ++j) {

				if (neigbs[j].pidx.isFluid()) {
					const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);
					h = (p_a.d + p_b.d) / 2;
					gradScalar = -ker_W_grad(neigbs[j].dis, h);
					grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

					p_a.VF_alpha2 += pow(p_b.restVolume, 2) * grad.dot(grad) / p_b.mass;
				}
				else if (neigbs[j].pidx.isCandidate()) {
					const CandidatePart& p_c = getCandidatePartOfIdx(neigbs[j].pidx);
					h = (p_a.d + p_c.d) / 2;
					gradScalar = -ker_W_grad(neigbs[j].dis, h);
					grad = (p_a.position - p_c.position) * (gradScalar / neigbs[j].dis);
				}
				else {
					const BoundPart& p_b = getBoundPartOfIdx(neigbs[j].pidx);
					h = (p_a.d + p_b.d) / 2;
					gradScalar = -ker_W_grad(neigbs[j].dis, h);
					grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);
				}
			}

			p_a.VF_alpha = p_a.VF_alpha2 / p_a.VF_gamma;
			if (p_a.VF_alpha < 1.0e-10) {
				p_a.VF_alpha = 1.0e-10;
			}
			p_a.VF_alpha = 1 / p_a.VF_alpha;
		}

	}
}

void IISph::VF_constantVolumeSolver() {

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
			p_a.restVolume = pow(p_a.d, vec_t::dim);
			p_a.advectionVelocity = p_a.velocity + p_a.advectionAcceleration * m_TH.dt;
		}

		while (errorRate > errorRateGoal || currentIteration < minimumIteration) {
			//cout << "constant volume iter: " << currentIteration << endl;
			if (currentIteration > maximumIteration) break;
			currentIteration++;

			//cout << "errorRate: " << errorRate << endl;

			real_t densitySum = 0;

			for (int n_b = int(m_Solids.size()), k = 0; k < n_b; ++k) {

				std::vector<BoundPart>& b_parts = m_Solids[k].boundaryParticles;
				const std::vector<NeigbStr>& b_neigbs = mg_NeigbOfSolids[k];
				int b_num = int(b_parts.size());

#pragma omp parallel for
				for (int i = 0; i < b_num; ++i) {
					BoundPart& p_a = b_parts[i];

					vec_t grad = vec_t::O;
					real_t gradScalar = 0;
					real_t h = p_a.d;

					p_a.VF_advectionVolume = p_a.VF_sphVolume;

					const Neigb* neigbs = b_neigbs[i].neigs; int n = b_neigbs[i].num;

					for (int j = 0; j < n; ++j) {
						if (neigbs[j].pidx.isFluid()) {
							const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);
							h = (p_a.d + p_b.d) / 2;
							gradScalar = -ker_W_grad(neigbs[j].dis, h);
							grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

							p_a.VF_advectionVolume += (grad.dot(p_a.velocity - p_b.advectionVelocity) * p_a.volume * p_b.restVolume) * m_TH.dt;
						}
						else if (neigbs[j].pidx.isCandidate()) {
							const CandidatePart& p_c = getCandidatePartOfIdx(neigbs[j].pidx);
							h = (p_a.d + p_c.d) / 2;
							gradScalar = -ker_W_grad(neigbs[j].dis, h);
							grad = (p_a.position - p_c.position) * (gradScalar / neigbs[j].dis);

							p_a.VF_advectionVolume += (grad.dot(p_a.velocity - p_c.velocity) * p_a.volume * p_c.volume) * m_TH.dt;
						}
						else {
							const BoundPart& p_b = getBoundPartOfIdx(neigbs[j].pidx);
							h = (p_a.d + p_b.d) / 2;
							gradScalar = -ker_W_grad(neigbs[j].dis, h);
							grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

							p_a.VF_advectionVolume += (grad.dot(p_a.velocity - p_b.velocity) * p_a.volume * p_b.volume) * m_TH.dt;
						}
					}

					if (p_a.VF_advectionVolume < p_a.volume) {
						p_a.VF_advectionVolume = p_a.volume;
					}

					p_a.VF_kappa = (p_a.VF_advectionVolume - p_a.volume) / pow(m_TH.dt, 2) / pow(p_a.volume, 3) * p_a.VF_alpha;
				}
			}

#pragma omp parallel for
			for (int i = 0; i < num; ++i) {
				FluidPart& p_a = f_parts[i];

				vec_t grad = vec_t::O;
				real_t gradScalar = 0;
				real_t h = p_a.d;

				p_a.VF_advectionVolume = p_a.VF_sphVolume;

				const Neigb* neigbs = f_neigbs[i].neigs; int n = f_neigbs[i].num;

				for (int j = 0; j < n; ++j) {
					if (neigbs[j].pidx.isFluid()) {
						const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);
						h = (p_a.d + p_b.d) / 2;
						gradScalar = -ker_W_grad(neigbs[j].dis, h);
						grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

						p_a.VF_advectionVolume += (grad.dot(p_a.advectionVelocity - p_b.advectionVelocity) * p_a.restVolume * p_b.restVolume) * m_TH.dt;
					}
					else if (neigbs[j].pidx.isCandidate()) {
						const CandidatePart& p_c = getCandidatePartOfIdx(neigbs[j].pidx);
						h = (p_a.d + p_c.d) / 2;
						gradScalar = -ker_W_grad(neigbs[j].dis, h);
						grad = (p_a.position - p_c.position) * (gradScalar / neigbs[j].dis);

						p_a.VF_advectionVolume += (grad.dot(p_a.advectionVelocity - p_c.velocity) * p_a.restVolume * p_c.volume) * m_TH.dt;
					}
					else {
						const BoundPart& p_b = getBoundPartOfIdx(neigbs[j].pidx);
						h = (p_a.d + p_b.d) / 2;
						gradScalar = -ker_W_grad(neigbs[j].dis, h);
						grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

						p_a.VF_advectionVolume += (grad.dot(p_a.advectionVelocity - p_b.velocity) * p_a.restVolume * p_b.volume) * m_TH.dt;
					}
				}

				if (p_a.VF_advectionVolume < p_a.restVolume) {
					p_a.VF_advectionVolume = p_a.restVolume;
				}
			}

#pragma omp parallel for
			for (int i = 0; i < num; ++i) { //particle "i" in "k"
				FluidPart& p_a = f_parts[i];
				p_a.VF_kappa = (p_a.VF_advectionVolume - p_a.restVolume) / pow(m_TH.dt, 2) / pow(p_a.restVolume, 3) * p_a.VF_alpha;
			}

			//calculate force
			for (int n_b = int(m_Solids.size()), k = 0; k < n_b; ++k) {
				if (m_Solids[k].dynamic) {
					if (m_Solids[k].type == Solid::RIGIDBODY) { // rigidbody
						std::vector<BoundPart>& b_parts = m_Solids[k].boundaryParticles;
						const std::vector<NeigbStr>& b_neigbs = mg_NeigbOfSolids[k];
						int b_num = int(b_parts.size());

#pragma omp parallel for
						for (int i = 0; i < b_num; ++i) {
							BoundPart& p_a = b_parts[i];
							vec_t grad = vec_t::O;
							real_t gradScalar = 0;
							real_t h = p_a.d;

							const Neigb* neigbs = b_neigbs[i].neigs; int n = b_neigbs[i].num;

							for (int j = 0; j < n; ++j) {
								if (neigbs[j].pidx.isFluid()) {
									const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);
									h = (p_a.d + p_b.d) / 2;
									gradScalar = -ker_W_grad(neigbs[j].dis, h);
									grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

									p_a.force = p_a.force - (grad *((pow(p_a.volume, 2) * p_b.restVolume * p_a.VF_kappa / p_a.VF_gamma)
											+ (pow(p_b.restVolume, 2) * p_a.volume * p_b.VF_kappa / p_b.VF_gamma)));
								}
								else if (neigbs[j].pidx.isCandidate()) {
									const CandidatePart& p_c = getCandidatePartOfIdx(neigbs[j].pidx);
									h = (p_a.d + p_c.d) / 2;
									gradScalar = -ker_W_grad(neigbs[j].dis, h);
									grad = (p_a.position - p_c.position) * (gradScalar / neigbs[j].dis);
								}
								else {
									const BoundPart& p_b = getBoundPartOfIdx(neigbs[j].pidx);
									h = (p_a.d + p_b.d) / 2;
									gradScalar = -ker_W_grad(neigbs[j].dis, h);
									grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

									p_a.force = p_a.force - (grad * ((pow(p_a.volume, 2) * p_b.volume * p_a.VF_kappa / p_a.VF_gamma)
										+ (pow(p_b.volume, 2) * p_a.volume * p_b.VF_kappa / p_b.VF_gamma)));
								}
							}
						}
					}
				}
			}
			
#pragma omp parallel for
			for (int i = 0; i < num; ++i) { //particle "i" in "k"
				FluidPart& p_a = f_parts[i];

				vec_t grad = vec_t::O;
				real_t gradScalar = 0;
				real_t h = p_a.d;

				const Neigb* neigbs = f_neigbs[i].neigs; int n = f_neigbs[i].num;

				for (int j = 0; j < n; ++j) {
					if (neigbs[j].pidx.isFluid()) {
						const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);
						h = (p_a.d + p_b.d) / 2;
						gradScalar = -ker_W_grad(neigbs[j].dis, h);
						grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

						//p_a.vel_adv = p_a.vel_adv - (grad * m_TH.dt / p_a.fm0 * ((pow(p_a.volume, 4) * p_a.VF_kappa / p_a.VF_volume) + (pow(p_b.volume, 4) * p_b.VF_kappa / p_b.VF_volume)));
						p_a.advectionVelocity = p_a.advectionVelocity - (grad * m_TH.dt / p_a.mass *
							((pow(p_a.restVolume, 2) * p_b.restVolume * p_a.VF_kappa / p_a.VF_gamma)
							+ (pow(p_b.restVolume, 2) * p_a.restVolume * p_b.VF_kappa / p_b.VF_gamma))
						);
					}
					else if (neigbs[j].pidx.isCandidate()) {
						const CandidatePart& p_c = getCandidatePartOfIdx(neigbs[j].pidx);
						h = (p_a.d + p_c.d) / 2;
						gradScalar = -ker_W_grad(neigbs[j].dis, h);
						grad = (p_a.position - p_c.position) * (gradScalar / neigbs[j].dis);

						//p_a.vel_adv = p_a.vel_adv - (grad * m_TH.dt / p_a.fm0 * (pow(p_c.volume, 2) * pow(p_a.volume, 2) * p_a.VF_kappa / p_a.VF_volume) * 2); // question
						p_a.advectionVelocity = p_a.advectionVelocity - (grad * m_TH.dt / p_a.mass
							* ((pow(p_a.restVolume, 2) * p_c.volume * p_a.VF_kappa / p_a.VF_gamma) * 2));
					}
					else {
						const BoundPart& p_b = getBoundPartOfIdx(neigbs[j].pidx);
						h = (p_a.d + p_b.d) / 2;
						gradScalar = -ker_W_grad(neigbs[j].dis, h);
						grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

						//p_a.vel_adv = p_a.vel_adv - (grad * m_TH.dt / p_a.fm0 * (pow(p_b.volume, 2) * pow(p_a.volume, 2) * p_a.VF_kappa / p_a.VF_volume) * 2); // question
						p_a.advectionVelocity = p_a.advectionVelocity - (grad * m_TH.dt / p_a.mass
							* ((pow(p_a.restVolume, 2) * p_b.volume * p_a.VF_kappa / p_a.VF_gamma)
								+ (pow(p_b.volume, 2) * p_a.restVolume * p_b.VF_kappa / p_b.VF_gamma)));
					}
				}

#pragma omp critical
				{sumErrorRate += (p_a.VF_advectionVolume - p_a.restVolume) / p_a.restVolume;}
			}

			errorRate = sumErrorRate / num;
			sumErrorRate = 0;
			//cout << "volume error rate: " << errorRate << endl;
		}
		//cout << "constant volume iter: " << currentIteration << endl;
		m_TH.cV_iteration += currentIteration - 1;
#pragma omp parallel for
		for (int i = 0; i < num; ++i) {
			FluidPart& p_a = f_parts[i];
			p_a.acce_presure = (p_a.advectionVelocity - p_a.velocity) / m_TH.dt;
		}

		//计算energy
		if (m_TH.enable_energy_computation) {
			if (m_TH.systemTime > m_TH.energy_tracing_frequency* m_TH.frequency_timer) {
				m_TH.potential_energy = 0;
				m_TH.kinetic_energy = 0;
				m_TH.sum_energy = 0;
				for (int i = 0; i < num; ++i) {
					FluidPart& p_a = f_parts[i];
					m_TH.potential_energy += p_a.mass * -m_TH.gravity_g[1] * (p_a.position[1] + 5);  // total potential energy
					m_TH.kinetic_energy += 0.5 * p_a.mass * p_a.velocity.dot(p_a.velocity); //total kinetic energy
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
			p_a.restVolume = pow(p_a.d, vec_t::dim);
			p_a.advectionVelocity = p_a.velocity + p_a.advectionAcceleration * m_TH.dt;
		}

		while (errorRate > errorRateGoal || currentIteration < minimumIteration) {
			//cout << "constant density iter: " << currentIteration << endl;
			if (currentIteration > maximumIteration) break;
			currentIteration++;

			real_t densitySum = 0;

			for (int n_b = int(m_Solids.size()), k_b = 0; k_b < n_b; ++k_b) {

				std::vector<BoundPart>& b_parts = m_Solids[k].boundaryParticles;
				const std::vector<NeigbStr>& b_neigbs = mg_NeigbOfSolids[k];

				int b_num = int(b_parts.size());

#pragma omp parallel for
				for (int i = 0; i < b_num; ++i) {

					BoundPart& p_a = b_parts[i];

					vec_t grad = vec_t::O; //gradient
					real_t gradScalar = 0; //gradient scalar version
					real_t h = p_a.d;
					p_a.advectionDensity = p_a.sphDensity;

					const Neigb* neigbs = b_neigbs[i].neigs;
					int n = b_neigbs[i].num;

					for (int j = 0; j < n; ++j) {
						if (neigbs[j].pidx.isFluid()) {
							const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);
							h = (p_a.d + p_b.d) / 2;
							gradScalar = -ker_W_grad(neigbs[j].dis, h);
							grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

							if (m_TH.adjustDensity) {
								p_a.advectionDensity += grad.dot(p_a.velocity - p_b.advectionVelocity) * p_a.mass * m_TH.dt;
							}
							else
							{
								p_a.advectionDensity += grad.dot(p_a.velocity - p_b.advectionVelocity) * p_b.mass * m_TH.dt;
							}
						}
						else if (neigbs[j].pidx.isCandidate()) {
							const CandidatePart& p_c = getCandidatePartOfIdx(neigbs[j].pidx);
							h = (p_a.d + p_c.d) / 2;
							gradScalar = -ker_W_grad(neigbs[j].dis, h);
							grad = (p_a.position - p_c.position) * (gradScalar / neigbs[j].dis);

							p_a.advectionDensity += grad.dot(p_a.velocity - p_c.velocity) * p_a.restDensity * p_c.volume * m_TH.dt;
						}
						else {
							const BoundPart& p_b = getBoundPartOfIdx(neigbs[j].pidx);
							h = (p_a.d + p_b.d) / 2;
							gradScalar = -ker_W_grad(neigbs[j].dis, h);
							grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

							p_a.advectionDensity += grad.dot(p_a.velocity - p_b.velocity) * p_b.restDensity * p_b.volume * m_TH.dt;
						}
					}

					if (p_a.advectionDensity < p_a.restDensity) { // when lack of neighbours, p_a should be regarded as restDensity
						p_a.advectionDensity = p_a.restDensity;
					}

					p_a.DF_kappa = (p_a.advectionDensity - p_a.restDensity) / pow(m_TH.dt, 2) * p_a.DF_alpha;
				}
			}

#pragma omp parallel for
			for (int i = 0; i < num; ++i) {

				FluidPart& p_a = f_parts[i];

				vec_t grad = vec_t::O; //gradient
				real_t gradScalar = 0; //gradient scalar version
				real_t h = p_a.d;
				p_a.advectionDensity = p_a.sphDensity;

				const Neigb* neigbs = f_neigbs[i].neigs; 
				int n = f_neigbs[i].num;

				for (int j = 0; j < n; ++j) {
					if (neigbs[j].pidx.isFluid()) {
						const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);
						h = (p_a.d + p_b.d) / 2;
						gradScalar = -ker_W_grad(neigbs[j].dis, h);
						grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

						if (m_TH.adjustDensity) {
							p_a.advectionDensity += grad.dot(p_a.advectionVelocity - p_b.advectionVelocity) * p_a.mass * m_TH.dt;
						}
						else
						{
							p_a.advectionDensity += grad.dot(p_a.advectionVelocity - p_b.advectionVelocity) * p_b.mass * m_TH.dt;
						}
					}
					else if (neigbs[j].pidx.isCandidate()) {
						const CandidatePart& p_c = getCandidatePartOfIdx(neigbs[j].pidx);
						h = (p_a.d + p_c.d) / 2;
						gradScalar = -ker_W_grad(neigbs[j].dis, h);
						grad = (p_a.position - p_c.position) * (gradScalar / neigbs[j].dis);

						p_a.advectionDensity += grad.dot(p_a.advectionVelocity - p_c.velocity) * p_a.restDensity * p_c.volume * m_TH.dt;
					}
					else {
						const BoundPart& p_b = getBoundPartOfIdx(neigbs[j].pidx);
						h = (p_a.d + p_b.d) / 2;
						gradScalar = -ker_W_grad(neigbs[j].dis, h);
						grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

						p_a.advectionDensity += grad.dot(p_a.advectionVelocity - p_b.velocity) * m_TH.boundaryDensity * p_b.volume * m_TH.dt;
					}
				}

				if (p_a.advectionDensity < p_a.restDensity) { // when lack of neighbours, p_a should be regarded as restDensity
					p_a.advectionDensity = p_a.restDensity;
				}

#pragma omp critical
				{ densitySum += (p_a.advectionDensity - p_a.restDensity) / p_a.restDensity; }
			}

#pragma omp parallel for
			for (int i = 0; i < num; ++i) {
				FluidPart& p_a = f_parts[i];
				p_a.DF_kappa = (p_a.advectionDensity - p_a.restDensity) / pow(m_TH.dt, 2) * p_a.DF_alpha;
			}

#pragma omp parallel for
			for (int i = 0; i < num; ++i) {
				FluidPart& p_a = f_parts[i];

				vec_t grad = vec_t::O;
				real_t gradScalar = 0;
				real_t h = p_a.d;

				const Neigb* neigbs = f_neigbs[i].neigs; int n = f_neigbs[i].num;

				for (int j = 0; j < n; ++j) {
					if (neigbs[j].pidx.isFluid()) {
						const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);
						h = (p_a.d + p_b.d) / 2;
						gradScalar = -ker_W_grad(neigbs[j].dis, h);
						grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

						p_a.advectionVelocity = p_a.advectionVelocity - (grad * p_b.mass * ((p_a.DF_kappa / p_a.sphDensity) + (p_b.DF_kappa / p_b.sphDensity)) * m_TH.dt);
					}
					else if (neigbs[j].pidx.isCandidate()) {
						const CandidatePart& p_c = getCandidatePartOfIdx(neigbs[j].pidx);
						h = (p_a.d + p_c.d) / 2;
						gradScalar = -ker_W_grad(neigbs[j].dis, h);
						grad = (p_a.position - p_c.position) * (gradScalar / neigbs[j].dis);

						p_a.advectionVelocity = p_a.advectionVelocity - (grad * p_a.restDensity * p_c.volume * ((p_a.DF_kappa / p_a.sphDensity)*2) * m_TH.dt);
					}
					else{
						const BoundPart& p_b = getBoundPartOfIdx(neigbs[j].pidx);
						h = (p_a.d + p_b.d) / 2;
						gradScalar = -ker_W_grad(neigbs[j].dis, h);
						grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

						p_a.advectionVelocity = p_a.advectionVelocity - (grad * p_b.restDensity * p_b.volume * ((p_a.DF_kappa / p_a.sphDensity) + (p_b.DF_kappa / p_b.sphDensity)) * m_TH.dt);
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
			p_a.acce_presure = (p_a.advectionVelocity - p_a.velocity) / m_TH.dt;
		}

		//计算energy
		if (m_TH.enable_energy_computation) {
			if (m_TH.systemTime > m_TH.energy_tracing_frequency* m_TH.frequency_timer) {
				m_TH.potential_energy = 0;
				m_TH.kinetic_energy = 0;
				m_TH.sum_energy = 0;
				for (int i = 0; i < num; ++i) {
					FluidPart& p_a = f_parts[i];
					m_TH.potential_energy += p_a.mass * -m_TH.gravity_g[1] * (p_a.position[1] + 5);  // total potential energy
					m_TH.kinetic_energy += 0.5 * p_a.mass * p_a.velocity.dot(p_a.velocity); //total kinetic energy
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
			p_a.restVolume = pow(p_a.d, vec_t::dim);
			p_a.advectionVelocity = p_a.velocity;
		}

		while (divergenceDeviationAver > errorRateGoal || currentIteration < minimumIteration) {
			//cout << "divergence free iter: " << currentIteration << endl;
			if (currentIteration > maximumIteration) break;
			currentIteration++;
			divergenceDeviationAver = 0;

			for (int b_f = int(m_Solids.size()), k_b = 0; k_b < b_f; ++k_b) {

				std::vector<BoundPart>& b_parts = m_Solids[k].boundaryParticles;
				const std::vector<NeigbStr>& b_neigbs = mg_NeigbOfSolids[k];

				int b_num = int(b_parts.size());

#pragma omp parallel for
				for (int i = 0; i < b_num; ++i) {
					BoundPart& p_a = b_parts[i];

					vec_t grad = vec_t::O;
					real_t gradScalar = 0;
					real_t h = p_a.d;
					p_a.DF_divergenceDeviation = 0;

					/* neighbour<vec> and its size */
					const Neigb* neigbs = b_neigbs[i].neigs; int n = b_neigbs[i].num;

					for (int j = 0; j < n; ++j) {
						if (neigbs[j].pidx.isFluid()) {
							const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);
							h = (p_a.d + p_b.d) / 2;
							gradScalar = -ker_W_grad(neigbs[j].dis, h);
							grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

							p_a.DF_divergenceDeviation += grad.dot(p_a.velocity - p_b.advectionVelocity) * p_b.mass;
						}
						else if (neigbs[j].pidx.isCandidate()) {
							const CandidatePart& p_c = getCandidatePartOfIdx(neigbs[j].pidx);
							h = (p_a.d + p_c.d) / 2;
							gradScalar = -ker_W_grad(neigbs[j].dis, h);
							grad = (p_a.position - p_c.position) * (gradScalar / neigbs[j].dis);

							p_a.DF_divergenceDeviation += grad.dot(p_a.advectionVelocity - p_c.velocity) * (p_a.restDensity * p_c.volume);
						}
						else {
							const BoundPart& p_b = getBoundPartOfIdx(neigbs[j].pidx);
							h = (p_a.d + p_b.d) / 2;
							gradScalar = -ker_W_grad(neigbs[j].dis, h);
							grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

							p_a.DF_divergenceDeviation += grad.dot(p_a.velocity - p_b.velocity) * (p_b.restDensity * p_b.volume);
						}
					}

					if (p_a.DF_divergenceDeviation < 0 || (p_a.sphDensity + p_a.DF_divergenceDeviation * m_TH.dt) < p_a.restDensity) {
						p_a.DF_divergenceDeviation = 0;
					}

					p_a.DF_kappaV = p_a.DF_divergenceDeviation * p_a.DF_alpha / m_TH.dt;
				}
			}

#pragma omp parallel for
			for (int i = 0; i < num; ++i) {
				FluidPart& p_a = f_parts[i]; 

				vec_t grad = vec_t::O;
				real_t gradScalar = 0;
				real_t h = p_a.d;
				p_a.DF_divergenceDeviation = 0;

				/* neighbour<vec> and its size */
				const Neigb* neigbs = f_neigbs[i].neigs; int n = f_neigbs[i].num;

				for (int j = 0; j < n; ++j) {
					if (neigbs[j].pidx.isFluid()) {
						const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);
						h = (p_a.d + p_b.d) / 2;
						gradScalar = -ker_W_grad(neigbs[j].dis, h);
						grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

						p_a.DF_divergenceDeviation += grad.dot(p_a.advectionVelocity - p_b.advectionVelocity) * p_b.mass;
					}
					else if (neigbs[j].pidx.isCandidate()) {
						const CandidatePart& p_c = getCandidatePartOfIdx(neigbs[j].pidx);
						h = (p_a.d + p_c.d) / 2;
						gradScalar = -ker_W_grad(neigbs[j].dis, h);
						grad = (p_a.position - p_c.position) * (gradScalar / neigbs[j].dis);

						p_a.DF_divergenceDeviation += grad.dot(p_a.advectionVelocity - p_c.velocity) * (p_a.restDensity * p_c.volume);
					}
					else {
						const BoundPart& p_b = getBoundPartOfIdx(neigbs[j].pidx);
						h = (p_a.d + p_b.d) / 2;
						gradScalar = -ker_W_grad(neigbs[j].dis, h);
						grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

						p_a.DF_divergenceDeviation += grad.dot(p_a.advectionVelocity - p_b.velocity) * (p_a.restDensity * p_b.volume);
					}
				}

				if (p_a.DF_divergenceDeviation < 0 || (p_a.sphDensity + p_a.DF_divergenceDeviation * m_TH.dt) < p_a.restDensity) {
					p_a.DF_divergenceDeviation = 0;
				}

				p_a.DF_kappaV = p_a.DF_divergenceDeviation * p_a.DF_alpha / m_TH.dt;
			}

#pragma omp parallel for
			for (int i = 0; i < num; ++i) { 
				FluidPart& p_a = f_parts[i];

				vec_t grad = vec_t::O;
				real_t gradScalar = 0;
				real_t h = p_a.d;

				const Neigb* neigbs = f_neigbs[i].neigs; int n = f_neigbs[i].num;

				for (int j = 0; j < n; ++j) {
					if (neigbs[j].pidx.isFluid()) {
						const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);
						h = (p_a.d + p_b.d) / 2;
						gradScalar = -ker_W_grad(neigbs[j].dis, h);
						grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

						p_a.advectionVelocity = p_a.advectionVelocity - (grad * p_b.mass * ((p_a.DF_kappaV / p_a.sphDensity) + (p_b.DF_kappaV / p_b.sphDensity)) * m_TH.dt);
					}
					else if (neigbs[j].pidx.isCandidate()) {
						const CandidatePart& p_c = getCandidatePartOfIdx(neigbs[j].pidx);
						h = (p_a.d + p_c.d) / 2;
						gradScalar = -ker_W_grad(neigbs[j].dis, h);
						grad = (p_a.position - p_c.position) * (gradScalar / neigbs[j].dis);

						p_a.advectionVelocity = p_a.advectionVelocity - (grad * p_a.restDensity * p_c.volume * ((p_a.DF_kappaV / p_a.sphDensity)*2) * m_TH.dt);
					}
					else {
						const BoundPart& p_b = getBoundPartOfIdx(neigbs[j].pidx);
						h = (p_a.d + p_b.d) / 2;
						gradScalar = -ker_W_grad(neigbs[j].dis, h);
						grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

						p_a.advectionVelocity = p_a.advectionVelocity - (grad * p_b.restDensity * p_b.volume * ((p_a.DF_kappaV / p_a.sphDensity) + (p_b.DF_kappaV / p_b.sphDensity)) * m_TH.dt);
					}
				}

#pragma omp critical
				{ divergenceDeviationAver += p_a.DF_divergenceDeviation; }
			}
			divergenceDeviationAver = divergenceDeviationAver / num * m_TH.dt;
			//cout << "divergence correction: " << divergenceDeviationAver << endl;
		}
		m_TH.dF_iteration += currentIteration - 1;
#pragma omp parallel for
		for (int i = 0; i < num; ++i) {
			FluidPart& p_a = f_parts[i];
			p_a.velocity = p_a.advectionVelocity;
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
			p_a.restVolume = pow(p_a.d, vec_t::dim);
			p_a.advectionVelocity = p_a.velocity;
		}

		while (divergenceDeviationAver > errorRateGoal || currentIteration < minimumIteration) {
			//cout << "divergence free iter: " << currentIteration << endl;
			if (currentIteration > maximumIteration) break;
			currentIteration++;

			//cout << "divergenceDeviationAver: " << divergenceDeviationAver << endl;
			divergenceDeviationAver = 0;

			for (int n_b = int(m_Solids.size()), k = 0; k < n_b; ++k) {

				std::vector<BoundPart>& b_parts = m_Solids[k].boundaryParticles;
				const std::vector<NeigbStr>& b_neigbs = mg_NeigbOfSolids[k];
				int num = int(b_parts.size());

#pragma omp parallel for
				for (int i = 0; i < num; ++i) {

					BoundPart& p_a = b_parts[i];

					vec_t grad = vec_t::O;
					real_t gradScalar = 0;
					real_t h = p_a.d;
					p_a.VF_divergenceDeviation = 0;

					const Neigb* neigbs = b_neigbs[i].neigs; int n = b_neigbs[i].num;

					for (int j = 0; j < n; ++j) {
						if (neigbs[j].pidx.isFluid()) {
							const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);
							h = (p_a.d + p_b.d) / 2;
							gradScalar = -ker_W_grad(neigbs[j].dis, h);
							grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

							p_a.VF_divergenceDeviation += grad.dot(p_a.velocity - p_b.advectionVelocity) * p_a.volume * p_b.restVolume;
						}
						else if (neigbs[j].pidx.isCandidate()) {
							const CandidatePart& p_c = getCandidatePartOfIdx(neigbs[j].pidx);
							h = (p_a.d + p_c.d) / 2;
							gradScalar = -ker_W_grad(neigbs[j].dis, h);
							grad = (p_a.position - p_c.position) * (gradScalar / neigbs[j].dis);

							p_a.VF_divergenceDeviation += grad.dot(p_a.velocity - p_c.velocity) * p_a.volume * p_c.volume;
						}
						else {
							const BoundPart& p_b = getBoundPartOfIdx(neigbs[j].pidx);
							h = (p_a.d + p_b.d) / 2;
							gradScalar = -ker_W_grad(neigbs[j].dis, h);
							grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

							p_a.VF_divergenceDeviation += grad.dot(p_a.velocity - p_b.velocity) * p_a.volume * p_b.volume;
						}
					}

					if (p_a.VF_divergenceDeviation < 0 || (p_a.VF_sphVolume + p_a.VF_divergenceDeviation * m_TH.dt) < p_a.volume) {
						p_a.VF_divergenceDeviation = 0;
					}

					p_a.VF_kappaV = p_a.VF_divergenceDeviation * p_a.VF_alpha / m_TH.dt / pow(p_a.volume, 3);
				}
			}

#pragma omp parallel for
			for (int i = 0; i < num; ++i) {

				FluidPart& p_a = f_parts[i];

				vec_t grad = vec_t::O;
				real_t gradScalar = 0;
				real_t h = p_a.d;
				p_a.VF_divergenceDeviation = 0;

				const Neigb* neigbs = f_neigbs[i].neigs; int n = f_neigbs[i].num;

				for (int j = 0; j < n; ++j) {
					if (neigbs[j].pidx.isFluid()) {
						const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);
						h = (p_a.d + p_b.d) / 2;
						gradScalar = -ker_W_grad(neigbs[j].dis, h);
						grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

						//p_a.VF_divergenceDeviation += grad.dot(p_a.vel_adv - p_b.vel_adv) * pow(p_b.volume, 2);
						p_a.VF_divergenceDeviation += grad.dot(p_a.advectionVelocity - p_b.advectionVelocity) * p_a.restVolume * p_b.restVolume;
					}
					else if (neigbs[j].pidx.isCandidate()) {
						const CandidatePart& p_c = getCandidatePartOfIdx(neigbs[j].pidx);
						h = (p_a.d + p_c.d) / 2;
						gradScalar = -ker_W_grad(neigbs[j].dis, h);
						grad = (p_a.position - p_c.position) * (gradScalar / neigbs[j].dis);

						//p_a.VF_divergenceDeviation += grad.dot(p_a.vel_adv - p_c.velocity) * pow(p_c.volume, 2);
						p_a.VF_divergenceDeviation += grad.dot(p_a.advectionVelocity - p_c.velocity) * p_a.restVolume * p_c.volume;
					}
					else {
						const BoundPart& p_b = getBoundPartOfIdx(neigbs[j].pidx);
						h = (p_a.d + p_b.d) / 2;
						gradScalar = -ker_W_grad(neigbs[j].dis, h);
						grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

						//p_a.VF_divergenceDeviation += grad.dot(p_a.vel_adv - p_b.velocity) * pow(p_b.volume, 2);
						p_a.VF_divergenceDeviation += grad.dot(p_a.advectionVelocity - p_b.velocity) * p_a.restVolume * p_b.volume;
					}
				}

				if (p_a.VF_divergenceDeviation < 0 || (p_a.VF_sphVolume + p_a.VF_divergenceDeviation * m_TH.dt) < p_a.restVolume) {
					p_a.VF_divergenceDeviation = 0;
				}

				p_a.VF_kappaV = p_a.VF_divergenceDeviation * p_a.VF_alpha / m_TH.dt / pow(p_a.restVolume, 3);
			}

			//calculate force
			for (int n_b = int(m_Solids.size()), k = 0; k < n_b; ++k) {
				if (m_Solids[k].dynamic) {
					if (m_Solids[k].type == Solid::RIGIDBODY) { // rigidbody
						std::vector<BoundPart>& b_parts = m_Solids[k].boundaryParticles;
						const std::vector<NeigbStr>& b_neigbs = mg_NeigbOfSolids[k];
						int b_num = int(b_parts.size());

#pragma omp parallel for
						for (int i = 0; i < b_num; ++i) {
							BoundPart& p_a = b_parts[i];
							vec_t grad = vec_t::O;
							real_t gradScalar = 0;
							real_t h = p_a.d;

							const Neigb* neigbs = b_neigbs[i].neigs; int n = b_neigbs[i].num;

							for (int j = 0; j < n; ++j) {
								if (neigbs[j].pidx.isFluid()) {
									const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);
									h = (p_a.d + p_b.d) / 2;
									gradScalar = -ker_W_grad(neigbs[j].dis, h);
									grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

									p_a.force = p_a.force - (grad * ((pow(p_a.volume, 2) * p_b.restVolume * p_a.VF_kappaV / p_a.VF_gamma)
										+ (pow(p_b.restVolume, 2) * p_a.volume * p_b.VF_kappaV / p_b.VF_gamma)));
								}
								else if (neigbs[j].pidx.isCandidate()) {
									const CandidatePart& p_c = getCandidatePartOfIdx(neigbs[j].pidx);
									h = (p_a.d + p_c.d) / 2;
									gradScalar = -ker_W_grad(neigbs[j].dis, h);
									grad = (p_a.position - p_c.position) * (gradScalar / neigbs[j].dis);
								}
								else {
									const BoundPart& p_b = getBoundPartOfIdx(neigbs[j].pidx);
									h = (p_a.d + p_b.d) / 2;
									gradScalar = -ker_W_grad(neigbs[j].dis, h);
									grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

									p_a.force = p_a.force - (grad * ((pow(p_a.volume, 2) * p_b.volume * p_a.VF_kappaV / p_a.VF_gamma)
										+ (pow(p_b.volume, 2) * p_a.volume * p_b.VF_kappaV / p_b.VF_gamma)));
								}
							}
						}
					}
				}
			}

#pragma omp parallel for
			for (int i = 0; i < num; ++i) {

				FluidPart& p_a = f_parts[i];

				vec_t grad = vec_t::O;
				real_t gradScalar = 0;
				real_t h = p_a.d;

				const Neigb* neigbs = f_neigbs[i].neigs; int n = f_neigbs[i].num;

				for (int j = 0; j < n; ++j) {
					if (neigbs[j].pidx.isFluid()) {
						const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);
						h = (p_a.d + p_b.d) / 2;
						gradScalar = -ker_W_grad(neigbs[j].dis, h);
						grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

						p_a.advectionVelocity = p_a.advectionVelocity - (grad * m_TH.dt / p_a.mass * ((pow(p_a.restVolume, 2) * p_b.restVolume * p_a.VF_kappaV / p_a.VF_gamma)
							+ (pow(p_b.restVolume, 2) * p_a.restVolume * p_b.VF_kappaV / p_b.VF_gamma)));
					}
					else if (neigbs[j].pidx.isCandidate()) {
						const CandidatePart& p_c = getCandidatePartOfIdx(neigbs[j].pidx);
						h = (p_a.d + p_c.d) / 2;
						gradScalar = -ker_W_grad(neigbs[j].dis, h);
						grad = (p_a.position - p_c.position) * (gradScalar / neigbs[j].dis);

						p_a.advectionVelocity = p_a.advectionVelocity - (grad * m_TH.dt / p_a.mass * ((pow(p_a.restVolume, 2) * p_c.volume * p_a.VF_kappaV / p_a.VF_gamma) * 2));
					}
					else {
						const BoundPart& p_b = getBoundPartOfIdx(neigbs[j].pidx);
						h = (p_a.d + p_b.d) / 2;
						gradScalar = -ker_W_grad(neigbs[j].dis, h);
						grad = (p_a.position - p_b.position) * (gradScalar / neigbs[j].dis);

						p_a.advectionVelocity = p_a.advectionVelocity - (grad * m_TH.dt / p_a.mass * ((pow(p_a.restVolume, 2) * p_b.volume * p_a.VF_kappaV / p_a.VF_gamma) 
							+ (pow(p_b.volume, 2) * p_a.restVolume * p_b.VF_kappaV / p_b.VF_gamma)));
					}
				}
#pragma omp critical
				{ divergenceDeviationAver += p_a.VF_divergenceDeviation; }
			}

			divergenceDeviationAver = divergenceDeviationAver / num * m_TH.dt;
			//cout << "divergence error rate: " << divergenceDeviationAver << endl;
		}
		//cout << "divergence free iter: " << currentIteration << endl;
		m_TH.dF_iteration += currentIteration - 1;
#pragma omp parallel for
		for (int i = 0; i < num; ++i) {
			FluidPart& p_a = f_parts[i];
			p_a.velocity = p_a.advectionVelocity;
		}
	}
}

// paritlce-particle interaction, [Mon92], [BT07]
inline void IISph::f2fAdvection(
	FluidPart& fa, const FluidPart& fb, const real_t& dis)
{
	if (dis == 0) { ++m_EC.zeroDis; return; }
	vec_t xab = fa.position - fb.position;
	real_t h = (fa.d + fb.d) / 2;
	vec_t grad = -xab * ker_W_grad(dis, h) / dis;
	

	// viscosity
	real_t kinematicViscosity = (fa.kinematicViscosity + fb.kinematicViscosity) / 2;
	real_t pro = (fa.velocity - fb.velocity).dot(xab);
	vec_t viscosityAcceleration = grad * 2 * (vec_t::dim + 2) * kinematicViscosity * (fb.restVolume / fb.VF_gamma) * pro / (dis * dis + 1e-6);
	fa.advectionAcceleration += viscosityAcceleration;
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
		real_t nu = 2 * alpha * m_TH.smoothRadius_h * m_TH.soundSpeed_cs / (fa.sphDensity + fb.sphDensity);
		real_t pi = -nu * pro / (dis * dis + real_t(0.01) * m_TH.smoothRadius_h * m_TH.smoothRadius_h);
		acce += grad * (-(fma + fmb) / 2 * pi);
	}
	if (acce) { xab *= acce; fa.advectionAcceleration += xab; }

}

// fluid-rigid coupling, [AIS*12]
inline void IISph::b2fAdvection(
	FluidPart& fa, const BoundPart& rb, const real_t& dis)
{
	if (dis == 0) { ++m_EC.zeroDis; return; }
	vec_t xab = fa.position - rb.position;
	real_t h = (fa.d + rb.d) / 2;
	vec_t grad = -xab * ker_W_grad(dis, h) / dis;

	// viscosity
	real_t kinematicViscosity = (fa.kinematicViscosity + rb.kinematicViscosity) / 2;
	real_t pro = (fa.velocity - rb.velocity).dot(xab);
	vec_t viscosityAcceleration = grad * 2 * (vec_t::dim + 2) * kinematicViscosity * (rb.volume / rb.VF_gamma) * pro / (dis * dis + 1e-6);
	fa.advectionAcceleration += viscosityAcceleration;
}

//see 11


void IISph::advectionStep() {
	// foreach fluid
	for (int n_f = int(m_Fluids.size()), k = 0; k < n_f; ++k) {
		std::vector<FluidPart>& f_parts = m_Fluids[k].fluidParticles;
		const std::vector<NeigbStr>& f_neigbs = mg_NeigbOfFluids[k];
		//real_t fm0 = rho0 * v0;
		real_t alpha = m_Fluids[k].viscosity_alpha;
		real_t gamma = m_Fluids[k].surfaceTension_gamma;
		int num = int(f_parts.size());
		// foreach particle
#pragma omp parallel for
		for (int i = 0; i < num; ++i) {
			FluidPart& p_a = f_parts[i];
			p_a.presure = 0;
			p_a.advectionAcceleration += m_TH.gravity_g;
		}

#pragma omp parallel for
		for (int i = 0; i < num; ++i) {
			FluidPart& p_a = f_parts[i];
			//p_a.acce_adv = vec_t::O;
			const Neigb* neigbs = f_neigbs[i].neigs; int n = f_neigbs[i].num;
			// forearch neighbour
			for (int j = 0; j < n; ++j) {
				if (neigbs[j].pidx.isFluid()) { // fluid neighbour
					const FluidPart& p_b = getFluidPartOfIdx(neigbs[j].pidx);
					int idx_b = neigbs[j].pidx.toFluidI();
					if (idx_b == k) {
						// the same fluid
						f2fAdvection(p_a, p_b, neigbs[j].dis);
					}
					else {
						// different fluid
						//注：这里的p_a.volums是错误的，之后用到different fluid再改
						real_t fmb = p_a.restVolume * m_Fluids[idx_b].restDensity_rho0;
						real_t b_alpha = m_Fluids[idx_b].viscosity_alpha;
						ii_fluidPartForceExceptPressure_fdiff(
							p_a, p_b, neigbs[j].dis, p_a.mass, fmb, (alpha + b_alpha) / 2);

					}
				}
				else if (neigbs[j].pidx.isCandidate()) {//see 11
					const CandidatePart& p_c = getCandidatePartOfIdx(neigbs[j].pidx);
					int idx_c = neigbs[j].pidx.toCandidateI();
					real_t r_alpha = m_Candidates[idx_c].viscosity_alpha;
					computeForceFromCandidateToFluidExceptPressure(p_a, p_c, neigbs[j].dis, p_a.restDensity, r_alpha);
				}
				else { // boundary neighbour
					const BoundPart& p_b = getBoundPartOfIdx(neigbs[j].pidx);
					int idx_b = neigbs[j].pidx.toSolidI();
					real_t r_alpha = m_Solids[idx_b].viscosity_alpha;
					b2fAdvection(p_a, p_b, neigbs[j].dis);
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
		* (-fb.mass * (fa.presure / (fa.sphDensity * fa.sphDensity) + fb.presure / (fb.sphDensity * fb.sphDensity)));
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
	real_t dalta_a = fa.sphDensity / fma, dalta_b = fb.sphDensity / fmb;
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
		acce += grad * (-frho0 * rb.volume * (fa.presure / (fa.sphDensity * fa.sphDensity) * 2));
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
		real_t nu = 2 * r_alpha * m_TH.smoothRadius_h * m_TH.soundSpeed_cs / (fa.sphDensity * 2);
		real_t pi = -nu * pro / (dis * dis + real_t(0.01) * m_TH.smoothRadius_h * m_TH.smoothRadius_h);
		acce += grad * (-frho0 * cb.volume * pi);
	}
	if (acce) { xab *= acce; fa.advectionAcceleration += xab; }
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
		acce += grad * (-frho0 * cb.volume * (fa.presure / (fa.sphDensity * fa.sphDensity) * 2));
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
		force += grad * (-fm0 * frho0 * ca.volume * (fb.presure / (fb.sphDensity * fb.sphDensity) * 2));
	// viscosity

	real_t pro = (ca.velocity - fb.velocity).dot(xab);
	if (pro < 0) {
		real_t nu = 2 * r_alpha * m_TH.smoothRadius_h * m_TH.soundSpeed_cs / (fb.sphDensity * 2);
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
		force += grad * (-fm0 * frho0 * ra.volume * (fb.presure / (fb.sphDensity * fb.sphDensity) * 2));
	// viscosity
	real_t pro = (ra.velocity - fb.velocity).dot(xab);
	if (pro < 0) {
		real_t nu = 2 * r_alpha * m_TH.smoothRadius_h * m_TH.soundSpeed_cs / (fb.sphDensity * 2);
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
			p_a.acce_presure = p_a.advectionAcceleration;
			
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