#include <iostream>
#include <cmath>
#include <vector>
#include <cstdio>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#define USE_ARPACK
#include "../include/blas_lapack_template.h"
#include "../include/complex_def.h" 
#include "../include/matsubara_green_function.h"
#include "../include/aim.h"
#include "../include/basis.h"
#include "../include/exactdiag.h"
#include "../include/interface_arpack.h"

namespace py = pybind11;

class python_wrapper_for_paramagnetic_order_ED_solver
{
public:
	python_wrapper_for_paramagnetic_order_ED_solver()
	: _aim(NULL), isParameterPrepared(false), isDataPrepared(false)
	{}

	python_wrapper_for_paramagnetic_order_ED_solver(const int NUM_BATH, const int NIWN, const double BETA, const int NEVSP, const double BETASP)
	: _NUM_BATH(NUM_BATH), _NIWN(NIWN), _BETA(BETA), _NEVSP(NEVSP), _BETASP(BETASP), _omega(NIWN), _aim(new AndersonImpurityModel<double>),isParameterPrepared(true), isDataPrepared(false)
	{
		for(int i=0;i<_NIWN;++i) _omega[i] = std::complex<double>(0,(2.*i + 1)*M_PI/_BETA);
		_aim->add_bath(new BathStd(_NUM_BATH));
		_aim->no_more_bath();
	}

	~python_wrapper_for_paramagnetic_order_ED_solver()
	{
		if(_aim != NULL) {
			delete _aim;
		}
	}

	void parameterSetup(const int NUM_BATH, const int NIWN, const double BETA, const int NEVSP, const double BETASP)
	{
		_NUM_BATH = NUM_BATH;
		_NIWN = NIWN;
		_BETA = BETA;
		_NEVSP = NEVSP;
		_BETASP = BETASP;
		_omega.resize(_NIWN);

		for(int i=0;i<_NIWN;++i) {
			_omega[i] = std::complex<double>(0,(2.*i + 1)*M_PI/_BETA);
		}

		if(_aim != NULL) {
			delete _aim;
		}

		_aim = new AndersonImpurityModel<double>;
		_aim->add_bath(new BathStd(_NUM_BATH));
		_aim->no_more_bath();

		isParameterPrepared = true;
		isDataPrepared = false;
	}

	py::array_t<dcomplex> run_for_matsubara_frequency(py::array_t<double> PVEC, const double U, const double MU)
	{
		assert(isParameterPrepared);

		auto pvec = PVEC.mutable_unchecked<1>();
		_aim->parametrize(&pvec(0));

		MatsubaraGreenFunction<1> G(_NIWN);
	
		LinearOperator<Fop,double> Hop;
		HnintAIM<double> Himp;
		Himp(0,0,0,0)=-MU;
		Himp(0,1,0,1)=-MU;
	
		_aim->add_Hnint(Hop,Himp);
		_aim->add_Hu(Hop,U); 
		_aim->add_Hbath(Hop);

		// creating basis of Anderson impurity model.
		BasisSet<Conserved_NM> basis(_aim->nflavor(),_aim->norbital());
	
		// creating instance of ed-solver.
		ExactDiag<Fop,double, BasisSet<Conserved_NM> > ed(Hop,basis);

		// solve (nevSP ,  betaSP)
		ed.solve(EigenSparse(_NEVSP),_BETASP);

		HwrapperSpinSector< HnintAIM<double> > Hc(0,Himp);

		ed.compute_green<ContinuedFraction>(_NIWN,&_omega[0],Hc.generator(_aim->fermi()),G);

		_density = ed.observe((_aim->fermi())->number(0,0));
		_double_occupancy = ed.observe((_aim->fermi())->double_occupancy(0));

		isDataPrepared = true;
		
		return py::array_t<dcomplex>(_NIWN, &G[0]);
	}

	py::array_t<dcomplex> run_for_real_frequency(py::array_t<double> PVEC, const double U, const double MU, const py::array_t<dcomplex> omega_in)
	{
		assert(isParameterPrepared);

		auto pvec  = PVEC.mutable_unchecked<1>();
		const auto omega = omega_in.unchecked<1>();

		const int NTAU = omega.shape(0);

		_aim->parametrize(&pvec(0));

		MatsubaraGreenFunction<1> G(_NIWN);
	
		LinearOperator<Fop,double> Hop;
		HnintAIM<double> Himp;
		Himp(0,0,0,0)=-MU;
		Himp(0,1,0,1)=-MU;
	
		_aim->add_Hnint(Hop,Himp);
		_aim->add_Hu(Hop,U); 
		_aim->add_Hbath(Hop);

		// creating basis of Anderson impurity model.
		BasisSet<Conserved_NM> basis(_aim->nflavor(),_aim->norbital());
	
		// creating instance of ed-solver.
		ExactDiag<Fop,double, BasisSet<Conserved_NM> > ed(Hop,basis);

		// solve (nevSP ,  betaSP)
		ed.solve(EigenSparse(_NEVSP),_BETASP);

		HwrapperSpinSector< HnintAIM<double> > Hc(0,Himp);

		ed.compute_green<ContinuedFraction>(NTAU,&omega(0),Hc.generator(_aim->fermi()),G);

		_density = ed.observe((_aim->fermi())->number(0,0));
		_double_occupancy = ed.observe((_aim->fermi())->double_occupancy(0));

		isDataPrepared = true;

		return py::array_t<dcomplex>(NTAU, &G[0]);
	}

	py::array_t<double> anderson_impurity_projection(const py::array_t<double> PVEC, const py::array_t<dcomplex> bath_g, const double MU, const double TOLERANCE) const
	{
		assert(isParameterPrepared);
		
		const auto wrap_pvec = PVEC.unchecked<1>();
		const auto wrap_g = bath_g.unchecked<1>();

		std::vector<double> pvec(_NUM_BATH*2);
		std::memcpy(&pvec[0],&wrap_pvec(0),sizeof(double)*_NUM_BATH*2);
		MatsubaraGreenFunction<1> gamma(_NIWN);

		for(int i=0;i<_NIWN;++i) {
			gamma[i] = _omega[i] + MU - 1./wrap_g(i);
		}

		HnintAIM<double> Himp;
		Himp(0,0,0,0)=-MU;
		Himp(0,1,0,1)=-MU;
		HwrapperSpinSector< HnintAIM<double> > Hc(0,Himp);

		_aim->parametrize(_NIWN,&_omega[0],gamma,Hc.layout(),&pvec[0],1,TOLERANCE,1);
		
		return py::array_t<double>(_NUM_BATH*2, &pvec[0]);
	}

	double get_density() const
	{
		assert(isDataPrepared);
		return _density;
	}

	double get_double_occupancy() const
	{
		assert(isDataPrepared);
		return _double_occupancy;
	}

private:
	int _NUM_BATH;
	int _NIWN; 
	double _BETA;
	int _NEVSP;
	double _BETASP;
	std::vector<dcomplex> _omega;
	AndersonImpurityModel<double>* _aim;

	bool isParameterPrepared;
	bool isDataPrepared;

	double _density;
	double _double_occupancy;
};


PYBIND11_MODULE(_ed_solver, m) {
	m.doc() = "...";
	py::class_<python_wrapper_for_paramagnetic_order_ED_solver>(m, "ed_solver")
	.def(py::init<>())
	.def(py::init<const int, const int, const double, const int, const double>())
	.def("parameterSetup",&python_wrapper_for_paramagnetic_order_ED_solver::parameterSetup)
	.def("run",&python_wrapper_for_paramagnetic_order_ED_solver::run_for_matsubara_frequency)
	.def("run",&python_wrapper_for_paramagnetic_order_ED_solver::run_for_real_frequency)
	.def("anderson_impurity_projection",&python_wrapper_for_paramagnetic_order_ED_solver::anderson_impurity_projection)
	.def("get_density",&python_wrapper_for_paramagnetic_order_ED_solver::get_density)
	.def("get_double_occupancy",&python_wrapper_for_paramagnetic_order_ED_solver::get_double_occupancy);
}
