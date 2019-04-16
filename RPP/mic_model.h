#ifndef MIC_MODEL_H
#define MIC_MODEL_H

#include <vector>

class CMicModel
{
public:
	CMicModel(  double t_value );
	~CMicModel();

public:
	int parameter_estimation( std::vector<int> citation_time );

	void check_log_likelihood( std::vector<int> const & citation_time );

	void compute_log_likelihood( std::vector<int> const & citation_time, double lambda, double mu, double sigma );

public:
	double get_mu();
	double get_sigma();
	double get_lambda();
    double calc(double asktime);
    double getasktime();

private:
	double gaussian_cumulative_distribution_function( double upper );

	inline double gaussian_density_distribtuion_fuction( double x );

	double compute_lambda();

	double compute_gradient_m();

	double compute_gradient_mu();


	double compute_gradient_sigma();

	double log_likelihood();

	double log_normal( double x, double mu, double sigma );

	void hessian_matrix( double & mu_mu, double & mu_sigma, double & sigma_sigma);

	void numerical_hessian_matrix( double & mu_mu, double & mu_sigma, double & sigma_sigma );

	void transform_hessian_matrix( double & mu_mu, double & mu_sigma, double & sigma_sigma );

	void line_search( double delta_mu, double delta_sigma, double & LL );

private:
	double m_m;
	double m_T;
	double m_lambda;
	double m_mu;
	double m_sigma;
    double asktime;

	std::vector<int> m_citations;
};

#endif
