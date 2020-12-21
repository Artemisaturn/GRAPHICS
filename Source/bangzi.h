//���ӽ�ˮ
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
	glm::vec3 wheel = glm::vec3(3.5f, 7.0f, 0.0f);
	glm::vec3 wheel2 = glm::vec3(-0.0f, -7.0f, 0.0f);
	glm::vec3 wheels = glm::vec3(0.0f, 5.25f, 0.0f);
	glm::vec3 wheel2s = glm::vec3(-0.0f, -5.25f, 0.0f);

public:
	int show_rigid_ID = -1; //��������Ҫ������ʾ�ĸ���ID�����ڵ���0��Ч
	double current_angle = 0;	// ��ǰ�������x��ת���Ƕ�
	const double PI = 3.1415926;
	virtual void setupScene()
	{
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		const real_t vis = 0.05f;
		m_TH.vis = vis;
		if (vec_t::dim == 3) {
			m_TH.gravity_g[1] = real_t(-9.8);
			m_TH.spacing_r = 0.4f;
			m_TH.spacing_r22 = pow(m_TH.spacing_r/2, 2);
			m_TH.spacing_r26 = pow(2 * m_TH.spacing_r, 6);
			m_TH.particle_volume = m_TH.spacing_r*m_TH.spacing_r*m_TH.spacing_r;
			m_TH.smoothRadius_h = 2 * m_TH.spacing_r;
			m_TH.dt = real_t(1.0)*m_TH.spacing_r / m_TH.soundSpeed_cs * 3;
			m_TH.dt2 = pow(m_TH.dt, 2);
			m_TH.h = 0.2f;
			m_TH.r = 1.0f;
			m_TH.bt = 1.0f;

			//vorticity refinement & micropolar seetings
			m_TH.inertia_tensor = pow(m_TH.spacing_r / 2, 2) * 0.4 * m_TH.particle_volume * 1000;
			m_TH.enable_vortex = 1; // 1:vorticity refinement 2: micropolar
			m_TH.nu = 0.05f; //micropolar & vorticity refinement
			m_TH.nu_t = 0.4f; //micropolar
			m_TH.zeta = 0.05f; //micropolar
			m_TH.alpha = 0.5; //vorticity refinement
			m_TH.rigid_ID = 0; //micropolar
			m_TH.rigid_center = vec_t(3.0f, 4.0f, 0.0f);

			//energy tracing
			m_TH.fluid_ID = 1; //Ҫ׷��������Һ��ID
			m_TH.zero_altitude = -2.0f; //�߶ȵ��������
			m_TH.enable_energy_computation = true; //�Ƿ�������׷��
			m_TH.energy_tracing_frequency = 1; //ͳ�Ƽ��, ������ÿ0.1sͳ��һ��
			m_TH.spaceMin.set(-8);
			m_TH.spaceMax.set(8); m_TH.spaceMax[1] = 12;

			vec_t container_size = vec_t(7.0f, 3.0f, 7.0f);
			vec_t container_size_up = vec_t(7.0f, 1.0f, 7.0f);
			// container, solid 0
			BoundPart bp0; bp0.color[0] = bp0.color[1] = bp0.color[2] = 0.4f; bp0.color[3] = 1;
			bp0.position = bp0.velocity = bp0.force = vec_t::O;
			vec_t rb = container_size;
			addRigidCuboid(rb, bp0, 0, vis, false,
				glm::translate(glm::vec3(0, 0, 0)));

			// water
			FluidPart fp0; fp0.velocity = vec_t::O; fp0.density = 1000;
			fp0.color[1] = fp0.color[2] = 0.3f; fp0.color[0] = 0.9f; fp0.color[3] = 1;
			vec_t lb(-0.7f), rt(0.3f); rt[1] = 2; lb[1] += 5.2f; rt[1] += 10.0f;
			addFluidCuboid(true, 0, -container_size + vec_t(m_TH.spacing_r),
				container_size_up - vec_t(m_TH.spacing_r), fp0, 1000, vis, 1);

			addRigidFromPLY("yuanzhu.ply", "yuanzhusample.ply",
				bp0, 0, vis, glm::translate(wheel));

			m_TH.gravity_g[1] = real_t(-9.8);

		}
		else {

		}

		logInfoOfScene();
	}

protected:
	virtual void stepEvent()
	{
		if (getSystemTime() < 4 && getSystemTime() > 0) {
			glm::mat4 rot;
			rot = glm::translate(glm::vec3(0, -1.5 * m_TH.dt, 0));
			addSolidTransform(1, rot);
		}
		/*if (getSystemTime() > 3.5&&getSystemTime() < 40) {
			glm::mat4 rot;
			rot = glm::translate(wheels) * glm::rotate(rot, 3 * m_TH.dt, glm::vec3(0.0f, 20.0f, 5.0f)) * glm::translate(wheel2s);
			addSolidTransform(1, rot);
		}*/
		if (getSystemTime() > 4 && getSystemTime() < 12) {
			glm::mat4 rot;
			const double deltaD = 1;
			int r = 3.5;
			float dx = r * cos(glm::radians(current_angle + deltaD)) - r * cos(glm::radians(current_angle));
			float dy = r * sin(glm::radians(current_angle + deltaD)) - r * sin(glm::radians(current_angle));
			current_angle += deltaD;
			rot = glm::translate(glm::vec3(dx, 0, dy));
			addSolidTransform(1, rot);
		}
		if (getSystemTime() < 16 && getSystemTime() > 12) {
			glm::mat4 rot;
			rot = glm::translate(glm::vec3(0, 1.5 * m_TH.dt, 0));
			addSolidTransform(1, rot);
		}
	}
	void generateWater1() {

		if (vec_t::dim == 3) {
			// water outlet, initial position is on yoz plane and velocity along x+
			// radius(float) = radius*spacing_r   // radius Ϊ�뾶����������2017/10/29 spacing_rĿǰ���Ϊ С���ֱ��
			static const int radius = 4; static const float initial_vx = 10.0f;
			static glm::mat4 transform =   //��ƽ�ƺ���ת��������translate �� rotate,��ת���� vec3(x,y,z)Ϊ�ᣬ��ת ĳ��!
				glm::translate(glm::vec3(0.1f, 4.6f, -0.25f)) * glm::rotate(3 * 3.1415926f / 2, glm::vec3(0, 0, 1));
			static glm::mat4 trans_inv = glm::affineInverse(transform);
			static std::vector<FluidPart> newpats;  //�µ�һ�ģ�����һ�����ӡ�static�ֲ���������һ������ʱnewpatsΪ�գ���һ�����н���ʱ��newpatsΪ����һȦˮ���ӵ�ģ�壬������Զפ���ڴ��С�
			if (newpats.size() == 0) {                //��һ�� ���е�����ʱ��newpats == 0 �� ���´���Ϊ�� ����һȦ�µ�����           
				FluidPart fp0; fp0.velocity = vec_t::O; fp0.density = m_Fluids[0].restDensity_rho0;
				fp0.color[1] = fp0.color[2] = 0.3f; fp0.color[0] = 0.9f; fp0.color[3] = 0.3f;
				glm::vec4 v0 = transform * glm::vec4(initial_vx, 0, 0, 0);
				for (int zi = -radius; zi <= radius; ++zi) {
					for (int yi = -radius; yi <= radius; ++yi) {
						vec_t pi(0, yi*m_TH.spacing_r, zi*m_TH.spacing_r);
						if (pi.length() <= radius * m_TH.spacing_r + m_TH.spacing_r*0.1f)  // ���� forѭ������� һ�������Σ���������Ҫ���ɸ�����������Ƕ�� Բ��ˮ����
						{
							glm::vec4 p = transform * glm::vec4(pi[0], pi[1], pi[2], 1);
							fp0.position = vec_t(p[0], p[1], p[2]);
							fp0.velocity = vec_t(v0[0], v0[1], v0[2]);
							newpats.push_back(fp0);
						}
					}
				}
			} // if( newpats.size()==0 )   //��һ�����н���֮��newpats�ͳ�Ϊ�˼���һ��ˮ���ӵ�ģ�壬����פ�ڴ棬��������Ϊ�������ý��������ͷ�

			static const float t_interval = 1.1f* m_TH.spacing_r / initial_vx;
			static int next = 0; static float restrict_dis = 0;
			if (getSystemTime() <= 2) {
				// restrict particles, the last two layers
				restrict_dis += initial_vx * m_TH.dt;
				std::vector<FluidPart>& fluid_ps = m_Fluids[0].fluidParticles;
				for (size_t i = 0; i < newpats.size(); ++i) {
					if (fluid_ps.size() - 1 < i) break;
					FluidPart& p = fluid_ps[fluid_ps.size() - 1 - i];
					glm::vec4 p_yoz = trans_inv * glm::vec4(p.position[0], p.position[1], p.position[2], 1);
					glm::vec4 v_yoz = trans_inv * glm::vec4(p.velocity[0], p.velocity[1], p.velocity[2], 0);
					if (p_yoz.x < restrict_dis) {
						p_yoz.x = restrict_dis;
						if (v_yoz.x < 0) v_yoz.x = 0;
						p_yoz = transform * p_yoz;
						v_yoz = transform * v_yoz;
						p.position.set(p_yoz[0], p_yoz[1], p_yoz[2]);
						p.velocity.set(v_yoz[0], v_yoz[1], v_yoz[2]);
					}
				}
				float dis2 = restrict_dis + m_TH.spacing_r;
				for (size_t i = newpats.size(); i < newpats.size() * 2; ++i) {
					if (fluid_ps.size() - 1 < i) break;
					FluidPart& p = fluid_ps[fluid_ps.size() - 1 - i];
					glm::vec4 p_yoz = trans_inv * glm::vec4(p.position[0], p.position[1], p.position[2], 1);
					glm::vec4 v_yoz = trans_inv * glm::vec4(p.velocity[0], p.velocity[1], p.velocity[2], 0);
					if (p_yoz.x < dis2) {
						p_yoz.x = dis2;
						if (v_yoz.x < 0) v_yoz.x = 0;
						p_yoz = transform * p_yoz;
						v_yoz = transform * v_yoz;
						p.position.set(p_yoz[0], p_yoz[1], p_yoz[2]);
						p.velocity.set(v_yoz[0], v_yoz[1], v_yoz[2]);
					}
				}
				// push_back fluid particles
				if (getSystemTime() >= next * t_interval) {
					addFluidParts(0, newpats);    //����ĳ��ѭ����ĳ��ʱ�䲽����ĳ��֡���з��ϣ��� ����һȦ��
					++next; restrict_dis = 0;
				}
				//std::cout << getNumFluidParts() << '\n';
			}
			else {
				next = int(getSystemTime() / t_interval + 0.5f);
				restrict_dis = m_TH.spacing_r;
			}
			static std::ofstream file_num_fluidpartiles(getLogFileName() + " numFluidPaticles.log");
			file_num_fluidpartiles.setf(std::ios::left);
			file_num_fluidpartiles.width(11); file_num_fluidpartiles << getFrameNumber() << ' ';
			file_num_fluidpartiles << getNumFluidParts() << '\n';
		}// if(vec_t::dim==3) */
	}
};

#endif //#ifndef _DEMO_MYSPH_H_