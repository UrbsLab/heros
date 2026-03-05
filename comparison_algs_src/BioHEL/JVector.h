/*
   This library was downloaded from: http://www.mike95.com

   This library is copyright.  It may freely be used for personal purposes 
   if the restriction listed below is adhered to.
       Author: Michael Olivero
       Email:  mike95@mike95.com

   //===============================
   //Start of Restriction Definition
   //===============================
   Anyone can have full use of the library provided they keep this complete comment
   with the source.  Also I would like to ask if any changes are made to the
   code for efficiency reasons, please let me know so I may look into your change and
   likewise incorporate it into library.  If a suggestion makes it into the library,
   your credits will be added to this information.

   Authors of Computer related books are welcome to include this source code as part
   of their publishing, provided credit the Author and make note of where the source
   code was obtained from: http://www.mike95.com
   //=============================
   //End of Restriction Definition
   //=============================

   Description:
   Visit http://www.mike95.com/c_plusplus/classes/JVector/

   Standard collection class.  It's public member functions
   are identical to the Java Vector public member functions (except for any Java specific
   Java related functions).

   //The following people have contributed to the solution
   //of bugs or additional features in this library
   //=====================================================
   //Carl Pupa, email: pumacat@erols.com
   //Adam Doppelt, email: amd@gurge.com
*/

#include <cstdlib>
#include <iostream>
#ifndef _JVECTOR_H_
#define _JVECTOR_H_

#include "M95_types.h"

using namespace std;

template <class Etype>
class JVector
{
public:
	JVector( UINT initialCapacity = 100, UINT capacityIncrement = 100 );
	JVector( const JVector& rhv );
	virtual ~JVector();

	//Inspectors (additional exception throwing inspectors below)
	//===========================================================
	UINT capacity() const;
	bool contains( const Etype &elem ) const;
	const Etype & firstElement() const;
	int indexOf( const Etype &elem ) const;
	bool isEmpty() const;
	const Etype & lastElement() const;
	int lastIndexOf( const Etype &elem ) const;
	UINT size() const;
	void copyInto( Etype* array ) const;

	//Modifiers (additional exception throwing inspectors below)
	//==========================================================
	void addElement( const Etype &obj );
	void ensureCapacity( UINT minCapacity );
	void removeAllElements();
	bool removeElement( const Etype &obj );
	void setSize( UINT newSize );
	void trimToSize();

	//Exceptions are thrown at run time with the following functions if
	//the index parameter is not within a valid range.  If the data
	//is uncertain (i.e. user inputed), then you should wrap these function
	//calls with the try/catch blocks and handle them appropriately.
	//===============================================
	Etype & elementAt( UINT index ) const;		  //inspector
	void insertElementAt( const Etype &obj, UINT index ); //modifier
	void removeElementAt( UINT index );				   //modifier
	void setElementAt( const Etype &obj, UINT index );	//modifier


	//C++ specific operations
	//=======================
	const Etype & operator[]( UINT index ) const;
	Etype & operator[]( UINT index );
	bool operator ==(const JVector& rhv);

protected:
	int min( UINT left, UINT right ) const;
	void verifyIndex( UINT index ) const;
	UINT m_size;
	UINT m_capacity;
	UINT m_increment;
	Etype** m_pData;
};

//===============================================================
//Implementation of constructor, destructor, and member functions
//Necessary location for appropriate template enstantiation.
//===============================================================
template <class Etype>
JVector<Etype>::JVector( UINT initialCapacity, UINT capacityIncrement ) 
{
	m_size = 0;
	m_capacity = initialCapacity;
	m_pData = new Etype*[ m_capacity ];
	m_increment = capacityIncrement;
}

template <class Etype>
JVector<Etype>::JVector( const JVector<Etype>& rhv )
{
	m_size = rhv.m_size;
	m_capacity = rhv.m_capacity;
	m_pData = new Etype*[ m_capacity ];
	m_increment = rhv.m_increment;

	for( UINT i = 0; i < m_size; i++ )
	{
		m_pData[i] = new Etype( *(rhv.m_pData[i]) );
	}
}

template <class Etype>
JVector<Etype>::~JVector()
{
	removeAllElements();
	delete [] m_pData;
}

template <class Etype>
UINT
JVector<Etype>::capacity() const
{
	return m_capacity;
}

template <class Etype>
bool
JVector<Etype>::contains( const Etype &elem ) const
{
	for ( UINT i = 0; i < m_size; i++ )
	{
		if ( *m_pData[i] == elem )
			return true;
	}

	return false;
}

template <class Etype>
void
JVector<Etype>::copyInto( Etype* array ) const
{
	for( UINT i = 0; i < m_size; i++ )
		array[i] = *m_pData[i];
}


template <class Etype>
Etype &
JVector<Etype>::elementAt( UINT index ) const
{
	verifyIndex( index );
	return *m_pData[index];
}

template <class Etype>
const Etype &
JVector<Etype>::firstElement() const
{
	if ( m_size == 0 )
	{
		//throw "Empty JVector Exception";
		cerr << "firstElement() called on empty JVector" << endl;
		exit(1);
	}

	return *m_pData[ 0 ];
}

template <class Etype>
int
JVector<Etype>::indexOf( const Etype &elem ) const
{
	for ( UINT i = 0; i < m_size; i++ )
	{
		if ( *m_pData[ i ] == elem )
			return i;
	}

	return -1;
}

template <class Etype>
bool
JVector<Etype>::isEmpty() const
{
	return m_size == 0;
}

template <class Etype>
const Etype &
JVector<Etype>::lastElement() const
{
	if ( m_size == 0 )
	{
		//throw "Empty JVector Exception"
		cerr << "lastElement() called on empty JVector" << endl;
		exit(1);
	}

	return *m_pData[ m_size - 1 ];
}

template <class Etype>
int
JVector<Etype>::lastIndexOf( const Etype &elem ) const
{
	//check for empty vector
	if ( m_size == 0 )
		return -1;

	UINT i = m_size;
	
	do
	{
		i -= 1;
		if ( *m_pData[i] == elem )
			return i;

	}
	while ( i != 0 );

	return -1;
}

template <class Etype>
UINT
JVector<Etype>::size() const
{
	return m_size;
}

template <class Etype>
void
JVector<Etype>::addElement( const Etype &obj )
{
	if ( m_size == m_capacity )
		ensureCapacity( m_capacity + m_increment );

	m_pData[ m_size++ ] = new Etype( obj );
}

template <class Etype>
void
JVector<Etype>::ensureCapacity( UINT minCapacity )
{
	if ( minCapacity > m_capacity )
	{
		UINT i;
		m_capacity = minCapacity;

		Etype** temp = new Etype*[ m_capacity ];

		//copy all the elements over upto newsize
		for ( i = 0; i < m_size; i++ )
			temp[i] = m_pData[i];

		delete [] m_pData;
		m_pData = temp;
	}
}

template <class Etype>
void
JVector<Etype>::insertElementAt( const Etype &obj, UINT index )
{
	verifyIndex( index );	//this will throw if true

	if ( m_size == m_capacity )
		ensureCapacity( m_capacity + m_increment);

	Etype* newItem = new Etype(obj);	//pointer to new item
	Etype* tmp;							//temp to hold item to be moved over.

	for( UINT i = index; i <= m_size; i++ )
	{
		tmp = m_pData[i];
		m_pData[i] = newItem;

		if ( i != m_size )
			newItem = tmp;
		else
			break;
	}

	m_size++;
}

template <class Etype>
void
JVector<Etype>::removeAllElements()
{
	//avoid memory leak
	for ( UINT i = 0; i < m_size; i++ )
		delete m_pData[i];

	m_size = 0;
}

template <class Etype>
bool
JVector<Etype>::removeElement( const Etype &obj )
{
	for ( UINT i = 0; i < m_size; i++ )
	{
		if ( *m_pData[i] == obj )
		{
			removeElementAt( i );
			return true;
		}
	}

	return false;
}

template <class Etype>
void
JVector<Etype>::removeElementAt( UINT index )
{
	verifyIndex( index );

	delete m_pData[ index ];

	for ( UINT i = index+1; i < m_size; i++ )
		m_pData[ i - 1 ] = m_pData[ i ];

	m_size--;
}

template <class Etype>
void
JVector<Etype>::setElementAt( const Etype &obj, UINT index )
{
	verifyIndex( index );

	*m_pData[ index ] = obj;
}

template <class Etype>
void
JVector<Etype>::setSize( UINT newSize )
{
	if ( newSize > m_capacity )
		ensureCapacity( newSize );
	else if ( newSize < m_size )
	{
		for( UINT i = newSize; i < m_size; i++ )
			delete m_pData[i];

		m_size = newSize;
	}
}

template <class Etype>
void
JVector<Etype>::trimToSize()
{
	if ( m_size != m_capacity )
	{
		Etype** temp = new Etype*[ m_size ];
		UINT i;

		for ( i = 0; i < m_size; i++ )
			temp[i] = m_pData[i];

		delete [] m_pData;

		m_pData = temp;
		m_capacity = m_size;
	}
}

template <class Etype>
int
JVector<Etype>::min( UINT left, UINT right ) const
{
	return left < right ? left : right;
}
  
template <class Etype>
void
JVector<Etype>::verifyIndex( UINT index ) const
{
	if ( index >= m_capacity )
	{
		//throw "Index Out Of Bounds";
		cerr << "Index Out Of Bounds";
		exit(1);
	}
}

template <class Etype>
bool
JVector<Etype>::operator==( const JVector<Etype>& rhv)
{
	int i;

	if(m_size!=rhv.m_size) return false;

	for(i=0;i<m_size;i++) {
		if(*m_pData[i] != *(rhv.m_pData[i])) return false;
	}
	return true;
}



template <class Etype>
const Etype &
JVector<Etype>::operator[]( UINT index ) const
{
	return elementAt( index );
}

template <class Etype>
Etype &
JVector<Etype>::operator[]( UINT index )
{
	verifyIndex( index );
	return *m_pData[ index ];
}

#endif
