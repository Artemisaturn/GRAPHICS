
#ifndef _DEMO_MYSPH_H_
#define _DEMO_MYSPH_H_


#ifdef PREDICTION_PCISPH
#include "PciSph.h"
#else
#ifdef IISPH
#include "IISph.h"
#else
#include "WcSph.h"
#endif
#endif


#ifdef PREDICTION_PCISPH
class mySPH : public PciSph {
#else
#ifdef IISPH
class mySPH : public IISph {
#else
class mySPH : public WcSph {
#endif
#endif
public:

	int num_timer = 0;
	long long startTime = 0;
	long long finishTime = 0;
	bool isHave = false;

	virtual void setupScene()
	{
		const real_t vis = 0.05f;
		if (vec_t::dim == 3) {
			m_TH.Kelvin = 273.15;
			m_TH.spacing_r = 0.3;
			m_TH.smoothRadius_h = 3 * m_TH.spacing_r;
			m_TH.dt = real_t(1.0) * m_TH.spacing_r / m_TH.soundSpeed_cs * 2;
			//m_TH.dt = 0.002;
			m_TH.h = m_TH.smoothRadius_h;
			m_TH.r = 0.2f;
			m_TH.bt = 1.0f;

			m_TH.airTemperature = 273.15 + 10;
			m_TH.applyCohesion = false;
			m_TH.applyAdhesion = false;
			m_TH.phaseTransition = false;
			m_TH.applyDensityFluctuation = false;

			m_TH.dF_iteration = 0;
			m_TH.cV_iteration = 0;
			m_TH.potential_energy = 0;
			m_TH.kinetic_energy = 0;

			m_TH.spaceMin.set(-8);
			m_TH.spaceMax.set(8); m_TH.spaceMax[1] = 12;

			BoundPart bp0; bp0.color[0] = bp0.color[1] = bp0.color[2] = 0.4f; bp0.color[3] = 1;
			bp0.position = bp0.velocity = bp0.force = vec_t::O;
			vec_t rb(4.0f); rb[1] = 3.0f;
			bp0.temperature = 50.0 + 273.15;
			bp0.d = m_TH.spacing_r * 1.5;
			addRigidCuboid(rb, bp0, 0, vis, false, glm::translate(glm::vec3(0, 0, 0)));

			CandidatePart cp0;
			real_t cp0Density = 800;
			cp0.density = cp0Density;
			cp0.temperature = cp0.temperatureNext = -5.0 + 273.15;

			// 添加流体，要不崩

			vec_t relativeDis = vec_t(0, 0, 0);
			vec_t relativeDis_0 = vec_t(-2, 0.2, 0);
			vec_t relativeDis_1 = vec_t(2, 0.2, 0);

			FluidPart fp0;
			fp0.velocity = vec_t::O;
			fp0.temperature = fp0.temperatureNext = 0 + m_TH.Kelvin;
			fp0.beta = -0.01 * (fp0.temperature - m_TH.Kelvin) + 1;
			fp0.beta = 1;
			fp0.density = 1000 * fp0.beta;
			fp0.d = m_TH.spacing_r * 1.5;
			fp0.i_color = 1;
			
			addFluidCuboid(true, 0, relativeDis_0 + vec_t(-2.0f + m_TH.spacing_r * 2, -3.0f + m_TH.spacing_r * 2, -4.0f + m_TH.spacing_r * 2),
				relativeDis_0 + vec_t(2.0f - m_TH.spacing_r * 2, 3.0f + m_TH.spacing_r * 2, 4.0f - m_TH.spacing_r * 2), fp0, fp0.density, 1.0f, 1);

			FluidPart fp1; fp1.velocity = vec_t::O;
			fp1.temperature = fp1.temperatureNext = 0 + m_TH.Kelvin;
			fp1.beta = -0.01 * (fp1.temperature - m_TH.Kelvin) + 1;
			fp1.beta = 0.05;
			fp1.density = 100 * fp1.beta;
			fp1.d = m_TH.spacing_r * 1;
			fp1.i_color = 2;
			
			addFluidCuboid(false, 0, relativeDis_1 + vec_t(-2.0f + m_TH.spacing_r * 2, -3.0f + m_TH.spacing_r * 2, -4.0f + m_TH.spacing_r * 2),
				relativeDis_1 + vec_t(2.0f - m_TH.spacing_r * 2, 3.0f + m_TH.spacing_r * 2, 4.0f - m_TH.spacing_r * 2), fp1, fp1.density, 1.0f, 1);

			/*
			int k = m_Fluids[0].fluidParticles.size() - 1;
			for (int i = 0; i < k; i++) {
				m_Fluids[0].fluidParticles.erase(m_Fluids[0].fluidParticles.begin());
			}
			*/
		}
		else {

		}

		logInfoOfScene();
	}

protected:
	virtual void stepEvent()
	{
		// 添加兔子
		/*if (getSystemTime() > 1.0f && isHave) {
			cout << "Time is :" << getSystemTime() << endl;

			const real_t vis = 0.05f;

			CandidatePart cp0;
			real_t cp0Density = 800;
			cp0.density = cp0Density;
			cp0.temperature = cp0.temperatureNext = -5.0 + 273.15;

			addIceFromPLY("bunny.ply", "bunny-sample.ply", 0, cp0, vis, glm::scale(glm::vec3(0.0f)), cp0Density, vec_t(0, 10, 0), true);
			isHave = false;
		}*/

	}

};

#endif //#ifndef _DEMO_MYSPH_H_
