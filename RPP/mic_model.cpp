#include "mic_model.h"
#include <iostream>
#include <fstream>
#include <ctime>
#include <cmath>
#include <cstdlib>
#include <iomanip>

using namespace std;

const double sqrt2pi = 2.506628274631000502415765284811;
const double sqrt2 = 1.4142135623730950488016887242097;
const double pi = 3.14159265358979323846;
const double sigma_min = 0.1;
const double g_squre_resolution = 1e-10;
const int max_iter = 500;

CMicModel::CMicModel( double t_value )
{
	m_T = t_value;
	m_m = 30.0;
};

CMicModel::~CMicModel()
{
}

/*
 *	get the value of parameter mu
 *
 */
double CMicModel::get_mu()
{
	return m_mu;
}

/*
 *	get the value of parameter sigma
 *
 */
double CMicModel::get_sigma()
{
	return m_sigma;
}

/*
 *	get the value of parameter lambda
 *
 */
double CMicModel::get_lambda()
{
	return m_lambda;
}

int CMicModel::parameter_estimation( vector<int> citation_time )
{
    asktime = citation_time[citation_time.size() - 1];
    citation_time.pop_back();
	if ( citation_time.empty() )
	{
		//cout << "lambda:\t" << 0 << endl;
		//cout << "mu:\t any value" << endl;
		//cout << "sigma:\t any value" << endl;
		return 0;
	}

	m_citations = citation_time;
    //m_citations会在很多地方都使用到，只是设置一个全局变量；带m开头的变量大多如此

	double gradient_mu;
	double gradient_sigma;
	double mu_mu;
	double mu_sigma;
	double sigma_sigma;

	// initializing seed for random function
	srand( (int)time(NULL) );

	// initializing parameters mu and lambda, computing the parameter lambda
    //correction: initializing sigma, not lamda
	m_mu = 6 + 1.0 * rand() / RAND_MAX;    // generate random value between 0.0 and 2.0;
	m_sigma = 1.0 + 0.5 * rand() / RAND_MAX; // generate random value between 0.01 and 1.01;

	double LL = log_likelihood();

	double g_square = 0.0;
	double delta_mu;
	double delta_sigma;
	int round;
	for ( round = 1; round < max_iter; ++round )
	{
		gradient_mu = compute_gradient_mu();
		gradient_sigma = compute_gradient_sigma();
		hessian_matrix( mu_mu, mu_sigma, sigma_sigma );
		transform_hessian_matrix(mu_mu, mu_sigma, sigma_sigma);
		delta_mu = mu_mu * gradient_mu + mu_sigma * gradient_sigma;
		delta_sigma = mu_sigma * gradient_mu + sigma_sigma * gradient_sigma;
		
		// update parameters mu, sigma and lambda
		line_search( delta_mu, delta_sigma, LL );
				
		// checking convergence
		g_square = gradient_mu * gradient_mu + gradient_sigma * gradient_sigma;
		if ( g_square < g_squre_resolution )
			break;

		//cout << "log likelihood: " << LL << "\t" << "g_square: " << g_square << endl;
	}

	//cout << "\nround: " << round << endl;
	//cout << "lambda:\t" << m_lambda << endl;
	//cout << "mu:\t" << m_mu << endl;
	//cout << "sigma:\t" << m_sigma << endl;

	return 0;
}

/*
 * compute the log likelihood for the given parameters: mu, sigma, lambda
 *
 */
double CMicModel::log_likelihood()
{
	if ( m_citations.empty() )
		return 0;

	m_lambda = compute_lambda();

	double LL = 0.0;
	
	LL += m_citations.size() * log( m_lambda ) - m_citations.size();

	for ( size_t ii = 0; ii < m_citations.size(); ++ii )
	{
		LL += log(ii+m_m);
		double xi = (log((double)m_citations[ii])-m_mu) / m_sigma;
		LL -= xi * xi * 0.5 + log(sqrt2pi*m_sigma*m_citations[ii] );
	}

	return LL;
}

/*
 * compute the gradient of parameter m
 *
 */
double CMicModel::compute_gradient_m()
{
	double gradient_m = 0.0;

	gradient_m = -m_lambda * gaussian_density_distribtuion_fuction( (log(m_T)-m_mu)/m_sigma );

	if ( m_citations.empty() )
		return gradient_m;

	for ( size_t ii = 0; ii < m_citations.size(); ++ii )
	{
		if ( fabs(ii+m_m) > 1.0e-6 )
			gradient_m += 1.0 / (ii+m_m);
		else
			gradient_m += 1.0e6;
	}

	return gradient_m;
}

/*
 * compute the gradient of mu
 *
 */
double CMicModel::compute_gradient_mu( )
{
	double gradient_mu = 0.0;

	gradient_mu = m_lambda * (m_citations.size()+m_m) * gaussian_density_distribtuion_fuction( (log(m_T)-m_mu)/m_sigma ) / m_sigma;

	if ( m_citations.empty() )
		return gradient_mu;

	double sum = 0.0;
	for ( size_t ii = 0; ii < m_citations.size(); ++ii )
	{
		double tmp = (log((double)m_citations[ii])-m_mu)/m_sigma;
		sum += tmp - m_lambda * gaussian_density_distribtuion_fuction( tmp );
	}
	sum /= m_sigma;

	gradient_mu += sum;

	return gradient_mu;
}

/*
 * compute the gradient of sigma
 *
 */
double CMicModel::compute_gradient_sigma()
{
	double gradient_sigma = 0.0;

	double tmp = (log(m_T)-m_mu) / m_sigma;

	gradient_sigma = (m_lambda * (m_citations.size()+m_m) * tmp * gaussian_density_distribtuion_fuction( tmp ) - m_citations.size()) / m_sigma;

	if ( m_citations.empty() )
		return gradient_sigma;

	double sum = 0.0;
	for ( size_t ii = 0; ii < m_citations.size(); ++ii )
	{
		tmp = ( log((double)m_citations[ii])-m_mu ) / m_sigma;
		sum += tmp * ( tmp - m_lambda * gaussian_density_distribtuion_fuction(tmp) );
	}
	sum /= m_sigma;

	gradient_sigma += sum;

	return gradient_sigma;
}

/*
 *  compute lambda by letting derivative equal to 0.
 *
 */
double CMicModel::compute_lambda()
{
	if ( m_citations.empty() )
		return 0.0;  // if there is no citation, lambda = 0.0

	double lambda = (m_citations.size()+m_m) * gaussian_cumulative_distribution_function( (log(m_T)-m_mu)/m_sigma );
	
	for ( size_t ii = 0; ii < m_citations.size(); ++ii )
	{
		lambda -= gaussian_cumulative_distribution_function( (log((double)m_citations[ii])-m_mu)/m_sigma );
	}
	
	return m_citations.size() / lambda;
}

/*
 * compute the value of Gaussian density function at the place of x
 *
 */
inline double CMicModel::gaussian_density_distribtuion_fuction( double x )
{
	return exp(-x*x/2.0) / sqrt2pi;
}

/*
 * the error function used to compute the cumulative Gasussian distribution function 
 *
 */
inline double erf (double x)
{
	// constants
	const double a1 =  0.254829592;
	const double a2 = -0.284496736;
	const double a3 =  1.421413741;
	const double a4 = -1.453152027;
	const double a5 =  1.061405429;
	const double p =   0.3275911;

	// A&S formula 7.1.26
	double t = 1.0/(1.0 + p*std::fabs(x));
	double y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);

	if (x < 0)
		return -y;
	return y;
}

/*
 *	compute the value of cumulative Gasussian distribution function at the place of x
 *
 */
inline double CMicModel::gaussian_cumulative_distribution_function( double x )
{
	return (1.0+erf(x/sqrt2))*0.5;
}


/*
 *	compute the value of log normal density function at the place of x, with the parameters mu and sigma
 *
 */
inline double CMicModel::log_normal( double x, double mu, double sigma )
{
	double t = (log(x)-mu) / sigma;
	double value = exp( -0.5*t*t ) / x / sqrt2pi / sigma;

	return value;
}

void CMicModel::hessian_matrix( double & mu_mu, double & mu_sigma, double & sigma_sigma)
{
	// compute second derivate with respect to \mu
	mu_mu = 0.0;
	double tmp = (log(m_T)-m_mu)/m_sigma;

	mu_mu += m_lambda * (m_citations.size()+m_m) * tmp * gaussian_density_distribtuion_fuction(tmp);
	double sum = 0.0;
	for ( size_t ii = 0; ii < m_citations.size(); ++ii )
	{
		tmp = (log((double)m_citations[ii])-m_mu)/m_sigma;
		sum += -1.0 - m_lambda * tmp * gaussian_density_distribtuion_fuction( tmp );
	}
	mu_mu += sum;
	mu_mu /= m_sigma * m_sigma;

	// compute second derivate with respect to \mu and \sigma
	mu_sigma = 0.0;
	tmp = (log(m_T)-m_mu)/m_sigma;
	mu_sigma = - m_lambda * (m_citations.size()+m_m) * gaussian_density_distribtuion_fuction(tmp) * (1-tmp*tmp);
	sum = 0.0;
	for ( size_t ii = 0; ii < m_citations.size(); ++ii )
	{
		tmp = (log((double)m_citations[ii])-m_mu)/m_sigma;
		sum += -2.0 * tmp + m_lambda * gaussian_density_distribtuion_fuction(tmp) * (1-tmp*tmp);
	}
	mu_sigma += sum;
	mu_sigma /= m_sigma * m_sigma;

	// compute second derivate with respect to \sigma

	sigma_sigma = 0.0;
	tmp = (log(m_T)-m_mu)/m_sigma;
	sigma_sigma = -m_lambda * (m_citations.size()+m_m)*gaussian_density_distribtuion_fuction(tmp) * tmp * (2-tmp*tmp)+m_citations.size();
	sum = 0.0;
	for ( size_t ii = 0; ii < m_citations.size(); ++ii )
	{
		tmp = (log((double)m_citations[ii])-m_mu)/m_sigma;
		sum += tmp * ( -3.0 * tmp + m_lambda * gaussian_density_distribtuion_fuction(tmp) * (2.0 - tmp*tmp) );
	}
	sigma_sigma += sum;
	sigma_sigma /= m_sigma * m_sigma;
}



/*
 *	transform the hessian matrix into the inverse of negative hessian matrix
 *
 */
void CMicModel::transform_hessian_matrix(double & a, double & b, double & c)
{
	/*
	 * eigen value 1 = 1/2 * (a+c-sqrt(a*a+4*b*b-2*a*c+c*c))
	 * eigen value 2 = 1/2 * (a+c+sqrt(a*a+4*b*b-2*a*c+c*c))
	 * eigen vector 1 = [-(-a+c+sqrt(a*a+4*b*b-2*a*c+c*c)) / (2*b),1}
	 * eigen vector 2 = [-(-a+c-sqrt(a*a+4*b*b-2*a*c+c*c)) / (2*b),1}
	 */

	double a_c = a - c;
	double ac = a + c;
	double sqrt_delta = sqrt(a_c*a_c + 4*b*b);
	
	// compute the eigenvalue of matrix [a, b; b, c]
	double eigen_val1 = 1.0 / (0.5 * (ac - sqrt_delta));
	double eigen_val2 = 1.0 / (0.5 * (ac + sqrt_delta));
	if ( eigen_val1 < 0 )
		eigen_val1 = -eigen_val1;
	if ( eigen_val2 < 0 )
		eigen_val2 = -eigen_val2;
	

	/*
	 * eigen vec 1 = [p1, p3];
	 * eigen vec 2 = [p2, p4];
	 */

	double p1 = (a_c - sqrt_delta);
	double p2 = (a_c + sqrt_delta);
	double p3 = 2*b;
	double p4 = p3;

	// normalization of vector 1
	double norm1 = sqrt(p1*p1 + p3*p3);
	p1 /= norm1;
	p3 /= norm1;
	
	// normalization of vector 1
	double norm2 = sqrt(p2*p2 + p4*p4);
	p2 /= norm2;
	p4 /= norm2;

	// inverse of diagonalization
	a = p1*p1*eigen_val1 + p2*p2*eigen_val2;
	b = p1*p3*eigen_val1 + p2*p4*eigen_val2;
	c = p3*p3*eigen_val1 + p4*p4*eigen_val2;
}

/*
 *	search the new value of mu and sigma which can increase the log likelihood, along the direction of gradient
 *
 */
void CMicModel::line_search( double delta_mu, double delta_sigma, double & LL )
{
	double old_LL = LL;
	double old_sigma = m_sigma;
	double old_mu = m_mu;
	
	// check the boundary of sigma
	if ( m_sigma + delta_sigma < sigma_min )
	{
		//cout << "boundary check\n";
		delta_mu *= sigma_min-m_sigma / delta_sigma; 
		delta_sigma = sigma_min-m_sigma; // delta_sigma *= (sigma_min-m_sigma) / delta_sigma
	}

	//double old_delta_sigma = delta_sigma;
	//double old_delta_mu = delta_mu;
	
	// guarantee the increase of log likelihood
	for ( int ii = 0; ii < 100; ++ii )
	{
		m_sigma = old_sigma + delta_sigma;
		m_mu = old_mu + delta_mu;
		LL = log_likelihood();
		if ( LL >= old_LL ) return;

		delta_mu /= 2.0;
		delta_sigma /= 2.0;
	}

	m_sigma = old_sigma;
	m_mu = old_mu;

	//cout << "backward line search\n";
	//delta_mu = old_delta_mu;
	//delta_sigma = old_delta_sigma;
	//// guarantee the increase of log likelihood
	//for ( int ii = 0; ii < 100; ++ii )
	//{
	//	m_sigma = old_sigma - delta_sigma;
	//	m_mu = old_mu - delta_mu;
	//	//cout << "LL: " << LL << endl;
	//	LL = log_likelihood();
	//	if ( LL >= old_LL ) {cout << "ii = " << ii + 1 << "\tgreater\n"; return;}

	//	cout << "ii = " << ii + 1 << "\tLL = " << setw(15) << setprecision(10) << LL << "\n";
	//	delta_mu /= 2.0;
	//	delta_sigma /= 2.0;
	//}
}

/*
 *	print the landscape of log likelihood with respect to parameter mu and sigma
 *
 */
void CMicModel::check_log_likelihood( vector<int> const & citation_time )
{
	m_citations = citation_time;

	ofstream outFile( "log_likelihood.txt");
	if ( outFile.fail() )
	{
		cout << "Can not open the file: 'log_likelihood.txt'" << endl;
		return;
	}

	double l_mu = 7.4;
	double u_mu = 7.5;
	double l_sigma = 1.6;
	double u_sigma = 1.7;
	for ( m_mu = l_mu; m_mu < u_mu; m_mu += 0.001 )
	{
		for ( m_sigma = l_sigma; m_sigma < u_sigma; m_sigma += 0.001 )
		{
			double LL = log_likelihood();

			outFile << m_mu << "\t" << m_sigma << "\t" << LL << "\n";
		}
	}	

	outFile.close();
}

/*
 *	compute the hessian matrix numerically, i.e., using the gradient function
 *
 */
void CMicModel::numerical_hessian_matrix( double & mu_mu, double & mu_sigma, double & sigma_sigma )
{
	double delta = 0.00001;

	// compute mu_mu
	double old_mu = m_mu;
	m_mu -= delta;
	double left_gradient_mu_delta = compute_gradient_mu();
	m_mu = old_mu + delta;
	double right_gradient_mu_delta = compute_gradient_mu();
	mu_mu = (right_gradient_mu_delta - left_gradient_mu_delta) / (2*delta);
	m_mu = old_mu;

	// compute mu_sigma
	old_mu = m_mu;
	m_mu -= delta;
	double left_gradient_delta = compute_gradient_sigma();
	m_mu = old_mu + delta;
	double right_gradient_delta = compute_gradient_sigma();
	mu_sigma = (right_gradient_delta-left_gradient_delta) / (2*delta);
	m_mu = old_mu;

	// compute sigma_sigma
	double old_sigma = m_sigma;
	m_sigma -= delta;
	double left_gradient_sigma_delta = compute_gradient_sigma();
	m_sigma = old_sigma + delta;
	double right_gradient_sigma_delta = compute_gradient_sigma();
	sigma_sigma = (right_gradient_sigma_delta - left_gradient_sigma_delta) / (2*delta);
	m_sigma = old_sigma;
}

void CMicModel::compute_log_likelihood( vector<int> const & citation_time, double lambda, double mu, double sigma )
{
	m_lambda = lambda;
	m_mu = mu;
	m_sigma = sigma;
	m_citations = citation_time;

	cout << "LL: " << log_likelihood() << endl;
}


double CMicModel::calc(double asktime)
{
    return m_m * (exp(m_lambda * gaussian_cumulative_distribution_function((log(asktime) - m_mu) / m_sigma)) - 1);
}


double CMicModel::getasktime()
{
    return asktime;
}
