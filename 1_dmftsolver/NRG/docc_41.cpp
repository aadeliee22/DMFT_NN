#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <complex>
#include <functional>
#include <string>
#include <cstring>
#include <assert.h>
#include <fftw3.h>
/* 
--- How to use fftw3 library:
-I/home/hyejin/fftw-3.3.9 -L/opt/fftw/lib -lfftw3
*/

typedef std::vector<double> ImTimeGreen;
typedef std::vector<std::complex<double> > ImFreqGreen;

enum FOURIER_TRANSFORM { FORWARD = -1, BACKWARD = 1 };
void fft_1d_complex(const std::complex<double> * input, std::complex<double> * output,
  const int N, const FOURIER_TRANSFORM FFTW_DIRECTION)
{
  fftw_complex * in = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)*N)),
    * out = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)*N));
  fftw_plan p = fftw_plan_dft_1d(N, in, out, FFTW_DIRECTION, FFTW_ESTIMATE);

  for(int i=0; i<N; ++i)
  {
    in[i][0] = input[i].real();
    in[i][1] = input[i].imag();
  }   

  fftw_execute(p); // repeat as needed 

  for(int i=0; i<N; ++i)
    output[i] = std::complex<double>(out[i][0], out[i][1]);

  fftw_destroy_plan(p);

  fftw_free(in);
  fftw_free(out);
}

void GreenInverseFourier(const ImFreqGreen & giwn, const double beta,
  const std::vector<double> & M, ImTimeGreen & gtau)
{
  // check whether mesh size satisfies the Nyquist theorem.
  assert(gtau.size()/2 >= giwn.size());
  std::vector<std::complex<double> > iwn(giwn.size());
  for (int n=0; n<giwn.size(); ++n)
    iwn[n] = std::complex<double>(0, (2*n+1)*M_PI/beta);
  std::vector<std::complex<double> > giwn_temp(gtau.size(), std::complex<double>(0, 0)), gtau_temp(gtau.size());
  // giwn_temp[i] is 0.0 for i >= giwn.size().
  
  for (int n=0; n<giwn.size(); ++n){
    giwn_temp[n] = giwn[n] - (M[0]/iwn[n] + M[1]/std::pow(iwn[n], 2) + M[2]/std::pow(iwn[n], 3));
    }
    
  fft_1d_complex(giwn_temp.data(), gtau_temp.data(), gtau.size()-1, FORWARD);
  
  for (int i=0; i<gtau.size()-1; ++i){
  	gtau_temp[i] += giwn_temp[gtau.size()-1];
  }
  
  for (int i=0; i<gtau.size()-1; ++i)
  {
    double tau = beta*i/(gtau.size()-1.0);
    gtau[i] = 2.0/beta*(std::exp(std::complex<double>(0.0, -M_PI*tau/beta))*gtau_temp[i]).real()
      - 0.5*M[0] + (tau/2.0-beta/4.0)*M[1] + (tau*beta/4.0 - std::pow(tau, 2)/4.0)*M[2];
  }
  std::complex<double> gedge(0, 0); // := G(beta)
  // giwn_temp[i] is 0.0 for i >= giwn.size().
  for (int n=0; n<giwn.size(); ++n)
    gedge -= giwn_temp[n];
  gedge *= 2.0/beta;
  gtau[gtau.size()-1] = gedge.real() - 0.5*M[0] + (beta/4.0)*M[1];
}

double density(const ImFreqGreen & G_wn, const double beta, const std::vector<double>& M)
{
	const int Niwn = G_wn.size();
	std::complex<double> iwn;
	std::complex<double> dens = 0;

	for(int n=0;n<Niwn;++n)
	{
		iwn = std::complex<double>(0,(2*n+1)*M_PI/beta);
		dens += (G_wn[n] - (M[0]/iwn + M[1]/std::pow(iwn,2) + M[2]/std::pow(iwn,3)));
	}
	dens *= 2./beta;
	dens += -0.5*M[0] + (-beta/4.)*M[1];

	return 0.-dens.real();
}


int main()
{
	
	int N = 80000;
	const double beta = 10000;
	int ntau = N*5;
	std::vector<double> w(N); 
	std::vector<double> M1(3);
	std::vector<double> M2 = {1,0,0.25};
	double G_real, G_imag, S_real, S_imag, I_real, I_imag;
	
	std::vector<std::complex<double> > G(N);
	std::vector<std::complex<double> > S(N);
	std::vector<std::complex<double> > I(N);
	std::vector<double> GT(ntau);
	
	std::string UU, lamb;
	std::cin >> UU >> lamb;
	double U = std::atof(UU.c_str());
	double mu = U/2.0;
	M1[0] = -0.5*U, M1[1] = 0.0, M1[2] = 0.0;
	std::ifstream filein1; filein1.open("./Giw_"+lamb+"/Giw_4to1_"+UU+".dat");
	if (filein1.is_open()){
		std::cout << std::fixed;
    	std::cout.precision(6);
		std::cout << "U= " << U << " ";
		for (int i = 0; i < N; i++) { 
			filein1 >> w[i] >> G_real >> G_imag;
			S_real = mu-G_real*(0.25+1/(G_real*G_real+G_imag*G_imag));
			S_imag = w[i]-G_imag*(0.25-1/(G_real*G_real+G_imag*G_imag));
			I_real = G_real*S_real - G_imag*S_imag;
			I_imag = G_real*S_imag + G_imag*S_real;
			G[i].real(G_real); G[i].imag(G_imag);
			S[i].real(S_real); S[i].imag(S_imag);
			I[i].real(I_real); I[i].imag(-I_imag);
		}
		
		GreenInverseFourier(I, beta, M1, GT);
		std::cout << "docc = " << GT[0]/U << " ";
		std::cout << "<n>  = " << density(G, beta, M2) << " " << std::endl;
	}
	else{
		std::cout << "Error, no input file found" << std::endl;
	}
	filein1.close();

	
	
	return 0;
}


