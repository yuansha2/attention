#include "mic_model.h"
#include <fstream>
#include <iostream>
#include <string>
#include <cstdlib>
#include <sstream>
#include<cmath>
using namespace std;

int training_time = 0;
int ask1 = 365 * 6, ask2 = 365 * 10;

int main(int argc, char **argv )
{
	//int trainYear = 10;
	vector< vector<int> > citations;

	ifstream inFile( "test.txt" );

	if ( inFile.fail() )
	{
		cout << "fail to open files.\n";
		exit(0);
	}
	string oneline;
	while ( getline(inFile,oneline) )
	{
		stringstream ss;

		//size_t pos = oneline.find('\t');
		//ss << oneline.substr(pos+1);
		ss << oneline;

		int cc, c;
		vector<int> citation;
        ss >> c;
		while ( ss >> cc )
		{
            citation.push_back( cc );
		}
        citation.push_back(c);
		citations.push_back( citation );
        //break;
	}

    /*
	int cc;
	vector<int> citation;
	while ( inFile >> cc )
	{
		if ( cc > 0 && cc <= training_time )
			citation.push_back( cc );
	}
	citations.push_back( citation );

	inFile.close();
    */
	
	
	ofstream outFile( "a.txt" );

	for ( size_t ii = 0; ii < citations.size(); ++ii )
	{
        int training_time = citations[ii][citations[ii].size() - 2];
        CMicModel model( training_time );
		model.parameter_estimation(citations[ii]);

        double out;
        int flag = 0;
        out = model.calc(model.getasktime());
        
        int cnt = 0;
        while (flag && cnt <= 20)
        {
            cnt++;
            CMicModel model2( training_time );
            model2.parameter_estimation(citations[ii]);
            flag = 0;
            out = model.calc(model.getasktime());
            if (isnan(out)) flag = 1;
        }
		cout << ii+1  << "\t" << model.getasktime() << "\t" <<  citations[ii].size() << "\t" << model.get_lambda() << "\t" << model.get_mu() << "\t" << model.get_sigma() << "\n";
        outFile << out << endl;
	}

	outFile.close();
	
	return 0;
}

// for robert and pie
//int main(int argc, char **argv )
//{
//	//string filename = "D:\\CCNR-Project\\PRL-1960to1969\\PhysRevLett-1960to1969\\";
//	ifstream inFile( "D:\\laszlo_citation_days.dat" );
//	if ( inFile.fail() )
//	{
//		cout << "Error: cannot open file." << endl;
//		return -1;
//	}
//
//	string oneline;
//	while (getline(inFile,oneline) )
//	{
//		stringstream ss;
//
//		ss << oneline;
//
//		vector<int> citation_time;
//		int ct;
//		const int T = 10 * 365;
//
//		int max_val = 0;
//		while ( ss >> ct )
//		{
//			if (ct <= 0)
//				continue;
//
//			if ( ct <= T )
//				citation_time.push_back(ct);
//		}
//
//		CMicModel model( T );
//		model.parameter_estimation(citation_time);
//
//		cout << citation_time.size() << "\t" << model.get_lambda() << "\t" << model.get_mu() << "\t" << model.get_sigma() << "\n";
//	}
//
//	
//
//	return 0;
//}
