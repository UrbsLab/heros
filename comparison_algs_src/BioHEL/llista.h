#ifndef _LLISTA_CONTENIDOR_

#define _LLISTA_CONTENIDOR_

#include <stdio.h>
#include <stdlib.h>

template <class X> class node
{
	private:
		X info;
		int clau;
		node *seg;

	public:
		node(X _info,int _clau) {info=_info;clau=_clau; seg=NULL;}
		node() {seg=NULL;}
		int get_clau() {return clau;}
		X get_element() {return info;}
		void modifica_element(X _info) {info=_info;}
		void set_successor(node *_seg) {seg=_seg;}
		node *get_successor() {return seg;}
};

template <class X> class llista
{
	private:
		node<X> *primer;
	public:
		llista();   
		~llista();
		void inserir(X element,int clau);
		void modificar(X element,int clau);
		void borrar(int clau);
		X consultar(int clau);
		int hi_ha_param(int clau);
};


template <class X> llista<X>::llista()
{
	primer=new node<X>;

	if(!primer) {
		perror("llista:llista:out of memory");
		exit(1);
	}
}

template <class X> llista<X>::~llista()
{
	node<X> *tmp;

	while(primer!=NULL) {
		tmp=primer;
		primer=primer->get_successor();
		delete tmp;
	}
}


template <class X> void llista<X>::inserir(X element,int clau)
{
	node<X> *tmp=new node<X>(element,clau);

	if(tmp==NULL) {
		perror("llista:inserir:out of memory");
		exit(1);
	}

	tmp->set_successor(primer->get_successor());
	primer->set_successor(tmp);

}


template <class X> void llista<X>::modificar(X element,int clau)
{
	node<X> *tmp;
	int trobat=0;

	tmp=primer->get_successor();

	while(!trobat && tmp!=NULL) {
		if(clau==tmp->get_clau()) {
			trobat=1;
			tmp->modifica_element(element);
		}
		else tmp=tmp->get_successor();
	}
	if(!trobat) {
		inserir(element,clau);
	}
}

template <class X> void llista<X>::borrar(int clau)
{
	node<X> *ant,*act;
	int trobat=0;

	ant=primer;
	act=primer->get_successor();

	while(!trobat && act!=NULL) {
		if(clau==act->get_clau) {
			trobat=1;
			ant->set_successor(act->get_successor());
			delete act;
		}
		else {
			ant=act;
			act=act->get_successor();
		}
	}
	if(!trobat) {
		fprintf(stderr,"llista:borrar:no trobat %d\n",clau);
		exit(1);
	}
	
}

template <class X> X llista<X>::consultar(int clau)
{
	node<X> *tmp;
	int trobat=0;
	X element;

	tmp=primer->get_successor();

	while(!trobat && tmp!=NULL) {
		if(clau==tmp->get_clau()) {
			trobat=1;
			element=tmp->get_element();
		}
		else tmp=tmp->get_successor();
	}
	if(!trobat) {
#ifndef _WINDOWS
		fprintf(stderr,"llista:consultar:no trobat %d\n",clau);
		exit(1);
#else
		CString m_str;
		m_str.Format("Llista:consultar:no trobat %d",clau);
		AfxMessageBox(m_str);
#endif
	}
	return element;
}


template <class X> int llista<X>::hi_ha_param(int clau)
{
#define TRUE 1
#define FALSE 0

	node<X> *tmp;
	int trobat=FALSE;	
	
	tmp=primer->get_successor();

	while(!trobat && tmp!=NULL) {
		if(clau==tmp->get_clau()) {
			trobat=TRUE;
		}
		else tmp=tmp->get_successor();
	}
	return trobat;
}


#endif	
