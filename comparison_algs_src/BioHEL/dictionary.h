#ifndef _DICTIONARY_H_
#define _DICTIONARY_H_

#include <stdio.h>
#include <stdlib.h>

template <class X> class node
{
	private:
		X content;
		int key;
		node *next;

	public:
		node(X _content,int _key) {content=_content;key=_key; next=NULL;}
		node() {next=NULL;}
		int getKey() {return key;}
		X getContent() {return content;}
		void updateContent(X _content) {content=_content;}
		void setNext(node *_next) {next=_next;}
		node *getNext() {return next;}
};

template <class X> class dictionary
{
	private:
		node<X> *first;

		void insertNew(X element,int key);
	public:
		dictionary();   
		~dictionary();
		void insertContent(X element,int key);
		void removeContent(int key);
		X getContent(int key);
		int keyExists(int key);
};


template <class X> dictionary<X>::dictionary()
{
	first=new node<X>;
}

template <class X> dictionary<X>::~dictionary()
{
	node<X> *tmp;

	while(first!=NULL) {
		tmp=first;
		first=first->getNext();
		delete tmp;
	}
}


template <class X> void dictionary<X>::insertNew(X element,int key)
{
	node<X> *tmp=new node<X>(element,key);
	tmp->setNext(first->getNext());
	first->setNext(tmp);
}


template <class X> void dictionary<X>::insertContent(X element,int key)
{
	node<X> *tmp;
	int found=0;

	tmp=first->getNext();

	while(!found && tmp!=NULL) {
		if(key==tmp->getKey()) {
			found=1;
			tmp->updateContent(element);
		}
		else tmp=tmp->getNext();
	}
	if(!found) {
		insertNew(element,key);
	}
}

template <class X> X dictionary<X>::getContent(int key)
{
	node<X> *tmp;
	int found=0;
	X element;

	tmp=first->getNext();

	while(!found && tmp!=NULL) {
		if(key==tmp->getKey()) {
			found=1;
			element=tmp->getContent();
		}
		else tmp=tmp->getNext();
	}
	if(!found) {
		fprintf(stderr,"dictionary:getContent:no found %d\n",key);
		fprintf(stderr,"Search configCodes.h for the meaning of the code\n");
		exit(1);
	}
	return element;
}

template <class X> void dictionary<X>::removeContent(int key)
{
	node<X> *tmp;
	int found=0;
	X element;

	tmp=first;

	while(tmp->getNext()!=NULL) {
		if(key==tmp->getNext()->getKey()) {
			node<X> *tmp2=tmp->getNext();
			tmp->setNext(tmp2->getNext());
			delete tmp2;
			return;
		}
		tmp=tmp->getNext();
	}
}



template <class X> int dictionary<X>::keyExists(int key)
{
#define TRUE 1
#define FALSE 0

	node<X> *tmp;
	int found=FALSE;	
	
	tmp=first->getNext();

	while(!found && tmp!=NULL) {
		if(key==tmp->getKey()) {
			found=TRUE;
		}
		else tmp=tmp->getNext();
	}
	return found;
}


#endif	
