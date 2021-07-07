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
  std::vector<std::complex<double> > giwn_temp(gtau.size(), std::complex<double>(0, 0)),
    gtau_temp(gtau.size());
  // giwn_temp[i] is 0.0 for i >= giwn.size().
  for (int n=0; n<giwn.size(); ++n)
    giwn_temp[n] = giwn[n] - (M[0]/iwn[n] + M[1]/std::pow(iwn[n], 2) + M[2]/std::pow(iwn[n], 3));
  fft_1d_complex(giwn_temp.data(), gtau_temp.data(), gtau.size()-1, FORWARD);	
  for (int i=0; i<gtau.size()-1; ++i)
  {
    const double tau = beta*i/(gtau.size()-1.0);
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

int main()
{
	int N = 5000;
	const double b = 100;
	int ntau = N*10;
	std::vector<double> w(N); 
	std::vector<double> M(3);
	M[0] = 1.0, M[1] = 0.0, M[2] = 0.0;
	double G_real, G_imag;
	
	std::vector<std::complex<double> > G(N);
	std::vector<double> GT(ntau);
	
	std::string U;
	U = "3.50";
	std::ifstream filein; filein.open("./Giw/Giw_1000_" + U + ".dat");
	if (filein.is_open()){
		std::cout << "Input file: U = " << U << std::endl;
		for (int i = 0; i < N; i++) { 
		filein >> w[i] >> G_real >> G_imag;
		G[i].real(G_real); G[i].imag(G_imag);
		}
		GreenInverseFourier(G, b, M, GT);
		std::ofstream fileout;
		fileout.open("./Gtau/Gt_1000_" + U + ".dat");
		for (int i = 0; i < ntau; i++){
			fileout << GT[i] << std::endl;
		}
		fileout.close();
	}
	else{
		std::cout << "Error, no input file found" << std::endl;
	}
	filein.close();
	
	return 0;
}


