/* Walk on triangle mesh.
*  https://arxiv.org/abs/2007.04940
*  https://kdocs.cn/l/ciGn4GHqHj8f
*  All rights reserved. Prometheus 2023.
*  Contributor(s): Neil Z. Shao.
*/
#pragma once
#include <Eigen/Eigen>

namespace prometheus
{
	struct TriangleWalkOption
	{
		// decay the remaining shift when across triangle
		// to avoid stack overflow
		double cross_triangle_decay = 0.9;
	};

	/* Point is given by indexed barycentric (p, v, w)
	*    p is the index of triangle
	*    (v, w) is the barycentric coordinates
	*  Shift the point by barycentric shift (delta_v, delta_w),
	*  solve for the result point on mesh (q, v', w')
	*/
	class TriangleWalk
	{
	public:
		/* a triangle with shifted index order
		*  for triangle ABC, AB should be the hypotenuse (shared edge for walk)
		*  B
		*  | \
		*  |  \
		*  |   \
		*  C -- A
		*  the barycenter coordinate for this ABC is
		*    a = coordinate along CA axis
		*    b = coordinate along CB axis
		*    c = 1 - a - b
		*/
		// point on mesh by indexed barycentric
		struct SurfacePoint
		{
			int f_idx = -1;
			Eigen::Vector3f bary;
		};

		// walking point
		struct WalkingPoint
		{
			Eigen::Vector2i edge_fi;
			Eigen::Vector2f intersect_ab;
			Eigen::Vector2f shift_ab;

			void finalize(SurfacePoint& spt, Eigen::Vector3f& shift);
		};

		TriangleWalk();
		~TriangleWalk();

		// init mesh
		void initTriangleMesh(const Eigen::MatrixXi& F);

		// walk on triangle mesh
		SurfacePoint walkSurfacePoint(SurfacePoint spt, Eigen::Vector3f shift);
		SurfacePoint walkCrossEdge(SurfacePoint spt, Eigen::Vector3f shift, int edge_idx);
		WalkingPoint walkToNeighbor(WalkingPoint wpt, Eigen::Vector2i nbr_edge);

		// helper functions
		auto& options() { return m_options; }
		auto& nbr_table() { return m_buffer.nbr_table; }
		void signalWalkingPoint(SurfacePoint spt, Eigen::Vector3f shift);
		auto& callback_walking_spt() { return m_callback_walking_spt; }

	private:
		TriangleWalkOption m_options;

		struct
		{
			// F is the triangle mesh faces
			Eigen::MatrixXi F;

			// triangle neighbor table
			// every triangle has 3 edges
			// every edge points to another triangle (with shifted vertex order)
			std::vector<std::array<Eigen::Vector2i, 3>> nbr_table;
		} m_buffer;

		// verbose callback
		std::function<void(SurfacePoint, Eigen::Vector3f)> m_callback_walking_spt = nullptr;
	};

}
