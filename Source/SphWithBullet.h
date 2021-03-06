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

#ifndef SPH_WITH_BULLET_H_
#define SPH_WITH_BULLET_H_


#include "SphBase.h"
#include "bt_inc.h"

#include "vcg_inc.h"

/*
 * integrate with bullet --------------------------------------------------------------------------
*/

class MBtSolid{

public:
	MBtSolid(btCollisionObject* btobj=0, GLTriMesh* trimesh=0 )
		: btObject(btobj), triMesh(trimesh) { }
	btCollisionObject* btObject;
	GLTriMesh* triMesh; // for rigidbody, use this to draw in opengl,
	// for softbody, use btSoftBody::m_faces
};

class SphWithBullet : public SphBase{

public:

	// Constructor
	SphWithBullet() { mbt_World.initialize(vect_to_btVect3(m_TH.gravity_g)); }
	
	// Destructor
	~SphWithBullet() {
		mbt_World.destroy();
		std::set<GLTriMesh*> ps; // avoid to delete non-null pointer already deleted
		for(size_t i=0; i<mbt_Solids.size(); ++i)
			ps.insert(mbt_Solids[i].triMesh);
		std::set<GLTriMesh*>::iterator it;
		for(it=ps.begin(); it!=ps.end(); ++it)
			if(*it) delete *it;
	}

protected:

	// add solid, set the data of mbt_World, mbt_Solids, m_Solids
	void addSolid( const Solid& sphsolid, const MBtSolid& mbtsolid );
	//see 9
	void addSolidForCandidate(const Solid& sphsolid, const MBtSolid& mbtsolid);

	// update solids using bullet,
	// synchronize boundary particles with rigid's transform or softbody's nodes
	void updateSolids();

	// for rigid: set the position and rotation, ignore indices.
	// for softbody: set nodes[indices[i]] = trans * innerPositions[indices[i]],
	// recall that innerPositions are initial positions of nodes,
	// if indices.size()==0, all nodes are processed.
	void setSolidTransform( int solidIdx,
		const glm::mat4& trans, const std::vector<int>& indices =std::vector<int>() );

	// for rigid: transform the rigid, WCS-> world coordinate system, ignore indices.
	// for softbody: set nodes[indices[i]] = trans * nodes[indices[i]], ignore isWCS,
	// if indices.size()==0, all nodes are processed.
	void addSolidTransform( int solidIdx,
		const glm::mat4& trans, bool isWCS=true, const std::vector<int>& indices =std::vector<int>() );


	// bullet Dynamics World
	class SphBulletWorld{
	public:
		// constructor
		SphBulletWorld() : dynamicsWorld(0),broadphase(0),dispatcher(0)
			,constraintSolver(0),collisionConfiguration(0) { }
		// Destructor
		~SphBulletWorld() {}

		// initialize bullet, should be called in the sph constructor
		void initialize(const btVector3& gravity);

		// bullet clear, i.e. delete bullet objects
		void destroy();

		// for btCompoundShape, push_back its children(can be compound!) to collisionShapes
		void pushBackShape(const btCollisionShape* shape);

		//std::vector<MeshShape> meshShapes; // triangle meshes and Collision Shapes
		//btDiscreteDynamicsWorld* dynamicsWorld;
		//btBroadphaseInterface* broadphase;
		//btCollisionDispatcher* dispatcher;
		//btSequentialImpulseConstraintSolver* constraintSolver;
		//btDefaultCollisionConfiguration* collisionConfiguration;
		btSoftRigidDynamicsWorld* dynamicsWorld;
		btAxisSweep3* broadphase;
		btCollisionDispatcher* dispatcher;
		btSequentialImpulseConstraintSolver* constraintSolver;
		btSoftBodyRigidBodyCollisionConfiguration* collisionConfiguration;
		std::vector<const btCollisionShape*> collisionShapes;
		btSoftBodyWorldInfo* softBodyWorldInfo;

	};

	SphBulletWorld mbt_World; // bullet Dynamics World data

	std::vector<MBtSolid> mbt_Solids; // correspond to m_Solids;
	

public:
	// convertion between btVector3 and vec_t and vec3_t
	static inline btVector3 vect_to_btVect3(const vec_t& vec){
		return btVector3( (btScalar)vec[0], (btScalar)vec[1], vec_t::dim==3?(btScalar)vec[2]:0 );
	}
	static inline vec_t btVec3_to_vect(const btVector3& btvec){
		vec_t re; re[0]=btvec[0]; re[1]=btvec[1]; if(vec_t::dim==3)re[2]=btvec[2]; return re;
	}
	static inline btVector3 vec3t_to_btVect3(const vec3_t& vec){
		return btVector3( (btScalar)vec[0], (btScalar)vec[1], (btScalar)vec[2] );
	}
	static inline vec3_t btVec3_to_vec3t(const btVector3& btvec){
		return vec3_t(btvec[0],btvec[1],btvec[2]);
	}

	void makeBtConvexHullShape(	btCollisionShape** cshape,
		const GLTriMesh& trimesh) const{
		float margin = float(m_TH.spacing_r/4);
		std::vector<float> vers;
		vcg_saveToData(trimesh, &vers, 0);
		*cshape = new btConvexHullShape( &vers[0], int(vers.size())/3, 3*sizeof(float) );
		(*cshape)->setMargin(margin);
	}

	void makeBtRigidBody( btRigidBody** rigid,
		btCollisionShape* cshape, float mass, const btTransform& trans) const{
		btVector3 localInertia(0,0,0);
		if(mass>0) cshape->calculateLocalInertia(mass,localInertia);
		btDefaultMotionState* myMotionState = new btDefaultMotionState(trans);
		*rigid = new btRigidBody( mass,myMotionState,cshape,localInertia );
		if(vec_t::dim==2){
			(*rigid)->setLinearFactor( btVector3(1, 1, 0) );
			(*rigid)->setAngularFactor( btVector3(0, 0, 1) );
		}
	}

	static void extractSoftBodyMeshData( const btSoftBody* soft,
		std::vector<float>* vers, std::vector<int>* tris ) {

		if( vers ){
			vers->clear();
			for(int i=0; i<soft->m_nodes.size(); ++i){
				vers->push_back(soft->m_nodes[i].m_x[0]);
				vers->push_back(soft->m_nodes[i].m_x[1]);
				vers->push_back(soft->m_nodes[i].m_x[2]);
			}
		}
		
		if( tris ){
			tris->clear();
			const btSoftBody::Node* ptr_begin = &soft->m_nodes[0];
			assert(  &soft->m_nodes[soft->m_nodes.size()-1]-ptr_begin == soft->m_nodes.size()-1 );
			//std::cout << "extractSoftBodyMeshData: " <<
			//(&soft->m_nodes[soft->m_nodes.size()-1]-ptr_begin==soft->m_nodes.size()-1) << '\n';
			for(int i=0; i<soft->m_faces.size(); ++i){
				tris->push_back(int(soft->m_faces[i].m_n[0]-ptr_begin));
				tris->push_back(int(soft->m_faces[i].m_n[1]-ptr_begin));
				tris->push_back(int(soft->m_faces[i].m_n[2]-ptr_begin));
			}
		}
		
	}

	void getSoftBodyMeshData( int idx,
		std::vector<float>* vers, std::vector<int>* tris ) const{
		assert( 0<=idx && idx<int(m_Solids.size()) );
		const btSoftBody* soft = btSoftBody::upcast(m_Solids[idx].mbtSolid_ptr->btObject); assert(soft);
		extractSoftBodyMeshData(soft, vers, tris);
	}

	void getRigidBodyTransform(int idx, glm::mat4& mat) const{
		assert( 0<=idx && idx<int(m_Solids.size()) );
		const btRigidBody* rigid = btRigidBody::upcast(m_Solids[idx].mbtSolid_ptr->btObject); assert(rigid);
		btTransform trans; rigid->getMotionState()->getWorldTransform(trans);
		trans.getOpenGLMatrix(&mat[0][0]);
	}


}; // class SphWithBullet


#endif // #ifndef SPH_WITH_BULLET_H_


