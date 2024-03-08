#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// for eigen support:
// must include <pybind11/eigen.h> before <Eigen/Eigen>
#include <pybind11/eigen.h>
#include "triangle_walk.cpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

// #define DEBUG_TRI_INNER

namespace py = pybind11;

/* Triwalk to walk on triangle mesh
*  - pass numpy: https://www.jianshu.com/p/c912a0a59af9
*/
class Triwalk
{
public:
    Triwalk(const Eigen::MatrixXi& F) 
    {
        if (F.rows() == 0) {
            throw std::runtime_error("[ERROR] F is empty");
        }
        if (F.cols() != 3) {
            throw std::runtime_error(std::string("[ERROR] F.cols() must be 3 != ") + std::to_string(F.cols()));
        }

        printf("[Triwalk] init mesh with F(%d, %d)\n", F.rows(), F.cols());
        m_triwalk.initTriangleMesh(F);
    }

    ~Triwalk() {}

    void operator()()
    {
        printf("[Triwalk]\n");
    }

    /* update surface points with triangle walk
    *  - { F, spt_vw }: index of triagnles and the barycentric coords within
    *  - spt_delta: update of barycentric coords
    */
    std::tuple<Eigen::VectorXi,Eigen::MatrixXd>
    updateSurfacePointsImpl(Eigen::VectorXi& spt_fidx, Eigen::MatrixXd& spt_vw, 
        Eigen::MatrixXd& spt_delta, bool verbose)
    {
#ifdef DEBUG_TRI_INNER
        if (verbose) {
            m_triwalk.callback_walking_spt() = [&](prometheus::TriangleWalk::SurfacePoint spt, Eigen::Vector3f shift) 
            {
                printf("[F = %d] bary = (%.4f, %.4f), shift = (%.4f, %.4f)\n", 
                    spt.f_idx, spt.bary[0], spt.bary[1], shift[0], shift[1]
                );
            };
        }
#endif

        for (int i = 0; i < spt_fidx.size(); ++i) {
            try {
                prometheus::TriangleWalk::SurfacePoint p_spt;
                p_spt.f_idx = spt_fidx[i];
                p_spt.bary << spt_vw(i, 0), spt_vw(i, 1), 1.0 - spt_vw(i, 0) - spt_vw(i, 1);
                Eigen::Vector3f shift;
                shift << spt_delta(i, 0), spt_delta(i, 1), 0.0 - spt_delta(i, 0) - spt_delta(i, 1);

                if (verbose) {
                    printf("[updateSurfacePoints][%d] f_idx = %d, bary = (%f, %f, %f)\n", i, spt_fidx[i], 
                        p_spt.bary[0], p_spt.bary[1], p_spt.bary[2]);
                    printf("[updateSurfacePoints]     shift = (%f, %f, %f)\n", shift[0], shift[1], shift[2]);
                }

                prometheus::TriangleWalk::SurfacePoint q_spt = m_triwalk.walkSurfacePoint(p_spt, shift);
                spt_fidx[i] = q_spt.f_idx;
                spt_vw(i, 0) = q_spt.bary[0];
                spt_vw(i, 1) = q_spt.bary[1];

                if (verbose) {
                    printf("[updateSurfacePoints]     => %d, (%f, %f, %f)\n", q_spt.f_idx, 
                        q_spt.bary[0], q_spt.bary[1], q_spt.bary[2]);
                }
            }
            catch (const char* msg) {
                std::cerr << "[updateSurfacePoints][ERROR] crash on [" << i << "] f_idx = "
                    << spt_fidx[i] << ", bary = " << spt_vw(i, 0) << ", " << spt_vw(i, 0) << std::endl;
                std::cerr << "[updateSurfacePoints][ERROR] " << msg << std::endl;
            }
        }

        return std::make_tuple(std::move(spt_fidx), std::move(spt_vw));
    }

    std::tuple<Eigen::VectorXi,Eigen::MatrixXd>
    updateSurfacePoints(Eigen::VectorXi& spt_fidx, Eigen::MatrixXd& spt_vw, 
        Eigen::MatrixXd& spt_delta)
    {
        return updateSurfacePointsImpl(spt_fidx, spt_vw, spt_delta, false);
    }

private:
    prometheus::TriangleWalk m_triwalk;
};


PYBIND11_MODULE(triwalk, m) {
    m.doc() = R"pbdoc(
        Walking on Triangle Mesh
        -----------------------
        - reference: https://arxiv.org/abs/2007.04940
        - init
        - update
    )pbdoc";

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    // verbose
    m.def("add", [](int i, int j) { return i + j; });

    // Triwalk
    py::class_<Triwalk>(m, "Triwalk", py::module_local())
    .def(py::init<const Eigen::MatrixXi&>())
    .def("updateSurfacePoints", &Triwalk::updateSurfacePoints)
    .def("updateSurfacePointsImpl", &Triwalk::updateSurfacePointsImpl)
    .def("__call__", &Triwalk::operator());

}
