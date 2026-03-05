#ifndef _CONFIG_MANAGEMENT_

#define _CONFIG_MANAGEMENT_

#include "dictionary.h"
#include "configCodes.h"

class configManagement
{
	private:
		dictionary<double> configuration;

	public: 
		inline double getParameter(int code) {
			return(configuration.getContent(code));
		}

		inline void removeParameter(int code) {
			configuration.removeContent(code);
		}

		inline void setParameter(double value,int code) {
			configuration.insertContent(value,code);
		}
	
		int thereIsParameter(int code) {
			return(configuration.keyExists(code));
		}
};

#endif
