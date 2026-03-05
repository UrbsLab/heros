#ifndef _MESSAGE_BUFFER_H_
#define _MESSAGE_BUFFER_H_

#define GRANULARITY 500000

#include <cstring>
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include "JVector.h"

class messageBuffer {
	JVector < char *>buffer;
	int currentSize;
	int enabled;
	int dumpToFile;
	int ignoreMessage;
	FILE *fp;

      public:
	 messageBuffer() {
		char *temp = new char[GRANULARITY];
		 temp[0] = 0;
		 buffer.addElement(temp);
		 currentSize = 0;
		 enabled = 0;
		 dumpToFile = 0;
		ignoreMessage=0;
	}

	~messageBuffer() {
		flushBuffer();
		delete buffer.elementAt(0);
		buffer.removeAllElements();
	}

	void ignoreMessages() {
		ignoreMessage=1;
	}
	void allowMessages() {
		ignoreMessage=0;
	}

	void setFile(char *fileName) {
		fp = fopen(fileName, "w");
		if (!fp) {
			fprintf(stderr, "Could not open log file\n");
			exit(1);
		}
		dumpToFile = 1;
	}

	inline void flushBuffer() {
		while (buffer.size() > 1) {
			char *temp = buffer.elementAt(0);
			if(dumpToFile) {
				fprintf(fp,"%s",temp);
				fflush(fp);
			} else {
				::printf("%s", temp);
				fflush(stdout);
			}
			delete temp;
			buffer.removeElementAt(0);
		}
		char *temp = buffer.elementAt(0);
		if(dumpToFile) {
			fprintf(fp,"%s",temp);
				fflush(fp);
		} else {
			::printf("%s", temp);
				fflush(stdout);
		}
		temp[0] = 0;
		currentSize = 0;
	}


	inline void addMessage(char *message) {
		if (!enabled) {
			if(dumpToFile) {
				fprintf(fp,"%s",message);
				fflush(fp);
			} else {
				::printf("%s", message);
				fflush(stdout);
			}
			return;
		}

		int length = strlen(message);
		while (currentSize + length >= GRANULARITY) {
			strncat(buffer.lastElement(), message,
				GRANULARITY - currentSize - 1);
			message += GRANULARITY - currentSize - 1;
			length -= GRANULARITY - currentSize - 1;
			char *temp = new char[GRANULARITY];
			temp[0] = 0;
			buffer.addElement(temp);
			currentSize = 0;
		}
		strcat(buffer.lastElement(), message);
		currentSize += length;
	}

	inline void printf(const char *fmt, ...) {

		if(ignoreMessage) return;
		/* Guess we need no more than 100 bytes. */
		int n, size = 1000;
		char *p;
		va_list ap;
		p = (char *) malloc(size);
		while (1) {
			va_start(ap, fmt);
			n = vsnprintf(p, size, fmt, ap);
			va_end(ap);

			if (n > -1 && n < size) {
				addMessage(p);
				free(p);
				return;
			}

			if (n < 0) {
				perror("vsnprintf failed");
				exit(1);
			}
			size = n + 1;
			p = (char *) realloc(p, size);
		}

	}



	inline void enable() {
		enabled = 1;
	}

	inline void disable() {
		flushBuffer();
		enabled = 0;
	}
};

#endif
