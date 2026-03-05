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
   Visit http://www.mike95.com/c_plusplus/classes/JHashtable/

   This class is a based on the Java JHashtable class and as such contains all the public
   member functions of it's Java equivalent.  Unlike Java, typecasts are not necessary
   since C++ allows template enstatiation of types at compile time.

   Note:  Since java has hashItem() as a member of the base Object class, all 
   Java classes are inheritly hashable.  Since the template parameter types do
   not necessarily have to have a built in hashing function, the user of the class
   must specify a hash function by calling setHashFunction() passing a pointer 
   to the hash function.

   The has function must be declared as the following:

       UINT function( const KeyType& );

   Where:
	function = any name you choose to use for the function name
	KeyType  = the type used for the key in the construction of the JHashtable object.

   Example:
       UINT myHash( const int& )
       {
           //your hashing code here for a key of type int.
       }


   //The following people have contributed to the solution
   //of bugs or additional features in this library
   //=====================================================
   //Jeremy Friesner, email: jaf@chem.ucsd.edu
   //Roxana Arama, email:roxanaa@tlc.ro

*/
#include "M95_types.h"
#include "JVector.h"

template <class KeyType, class ObjType>
class JHashtable
{
public:
	JHashtable( UINT initialCapacity = 101, float loadFactor = 0.5f );
	virtual ~JHashtable();

	//Inspectors
	//==========
	UINT size() const { return m_count; }
	bool isEmpty() const { return m_count == 0; }
	bool contains( const ObjType& value ) const;
	bool containsKey( const KeyType& key ) const;
	const ObjType& get( const KeyType& key );
	JVector<KeyType> keys();


	//Modifiers
	//=========
	void rehash();
	ObjType put( const KeyType& key, const ObjType& value );
	ObjType remove( const KeyType& key );
	void clear();

	//C++ specific user hash function
	void setHashFunction( UINT(*func)(const KeyType& key) ) { UserHash = func; }

private:
	//type for Entries
	struct JHashtableEntry
	{
		UINT hash;
		KeyType key;
		ObjType value;
		JHashtableEntry* next;
	};

	//Member variables
	UINT m_count;	//the size of the elements in the hashtable
	UINT m_tableSize;	//the size of the table.
	UINT m_threshold;
	float m_loadFactor;
	JHashtableEntry** m_table;

	
	//Helper functions
	UINT nextPrime( UINT start );
	UINT (*UserHash)( const KeyType& key );

	ObjType NULL_ITEM;	//used for returns of not found

};


//===============================================================
//Implementation of constructor, destructor, and member functions
//Necessary location for appropriate template enstantiation.
//===============================================================
template <class KeyType, class ObjType>
JHashtable<KeyType,ObjType>::JHashtable( UINT initialCapacity, float loadFactor )
{
	if ( loadFactor <= 0.0 )
	{
		cerr << "Bad Argument for loadFactor" << endl;
		exit(1);
	}

	m_count = 0;
	m_tableSize = initialCapacity;
	m_loadFactor = loadFactor;
	m_threshold = (UINT)(initialCapacity * loadFactor);

	//initialize and set to null
	//==========================
	m_table = new JHashtableEntry*[m_tableSize];
	for( UINT i = 0; i < m_tableSize; i++ )
		m_table[i] = NULL;

	//nullify the user hash function pointer
	UserHash = NULL;
}

template <class KeyType, class ObjType>
JHashtable<KeyType,ObjType>::~JHashtable( )
{
	clear();
	delete [] m_table;
}

template <class KeyType, class ObjType>
bool
JHashtable<KeyType,ObjType>::contains( const ObjType& value ) const
{
	JHashtableEntry** tab = m_table;
	for( UINT i = 0; i < m_tableSize; i++ )
	{
		for( JHashtableEntry* e = tab[i]; e != NULL; e = e->next )
		{
			if ( e->value == value )
				return true;
		}
	}

	return false;
}

template <class KeyType, class ObjType>
bool
JHashtable<KeyType,ObjType>::containsKey( const KeyType& key ) const
{
	JHashtableEntry** tab = m_table;
	UINT hash = (*UserHash)(key);
	
	UINT index = hash % m_tableSize;

	for( JHashtableEntry* e = tab[index]; e != NULL; e = e->next )
	{
		if ( e->hash == hash && e->key == key )
			return true;
	}

	return false;
}

template <class KeyType, class ObjType>
const ObjType&
JHashtable<KeyType,ObjType>::get( const KeyType& key )
{
	JHashtableEntry** tab = m_table;
	UINT hash = (*UserHash)(key);
	UINT index = hash % m_tableSize;

	for( JHashtableEntry* e = tab[index]; e != NULL; e = e->next )
	{
		if ( e->hash == hash && e->key == key )
			return e->value;
	}
	
	return NULL_ITEM; //if not found
}

template <class KeyType, class ObjType>
void
JHashtable<KeyType,ObjType>::rehash()
{
	UINT oldTableSize = m_tableSize;
	JHashtableEntry** oldTable = m_table;

	UINT newTableSize = nextPrime( 2 * oldTableSize );
	JHashtableEntry** newTable = new JHashtableEntry*[newTableSize];
	UINT i;
	for( i = 0; i < newTableSize; i++ )
		newTable[i] = NULL;

	m_threshold = (UINT)(newTableSize * m_loadFactor);
	m_table = newTable;

	//copy all the entries from the old to the new table
	for( i = 0; i < oldTableSize; i++ )
	{
		for( JHashtableEntry* old = oldTable[i]; old != NULL; )
		{
			JHashtableEntry* e = old;
			old = old->next;

			UINT index = e->hash % newTableSize;
			e->next = newTable[index];
			newTable[index] = e;
		}
	}

	m_tableSize = newTableSize;

	delete [] oldTable;
}

template <class KeyType, class ObjType>
ObjType
JHashtable<KeyType,ObjType>::put( const KeyType& key, const ObjType& value )
{
	JHashtableEntry** tab = m_table;
	UINT hash = (*UserHash)(key);
	UINT index = hash % m_tableSize;

	//Makes sure the key is not already in the hashtable
	//replace item if found
	JHashtableEntry* e;
	for( e = tab[index]; e != NULL; e = e->next )
	{
		if ( (e->hash == hash) && e->key == key )
		{
			ObjType old = e->value;
			e->value = value;
			return old;
		}
	}

	//Rehash the table if the threshold is exceeded
	if ( m_count >= m_threshold )
	{
		rehash();
		return put(key, value);
	}

	//Creates the new entry here
	e = new JHashtableEntry();
	e->hash = hash;
	e->key = key;
	e->value = value;
	e->next = tab[index];
	tab[index] = e;
	m_count++;

	return NULL_ITEM;
}

template <class KeyType, class ObjType>
ObjType
JHashtable<KeyType,ObjType>::remove( const KeyType& key )
{
	JHashtableEntry** tab = m_table;
	UINT hash = (*UserHash)(key);
	UINT index = hash % m_tableSize;
	
	for( JHashtableEntry* e = tab[index], *prev = NULL; e != NULL; prev = e, e = e->next)
	{
		if ( e->hash == hash && e->key == key )
		{
			if ( prev != NULL )
				prev->next = e->next;
			else
				tab[index] = tab[index]->next;

			ObjType theReturn = e->value;
			delete e;
			m_count--;	//decrement counter
			return theReturn;
		}
	}

	return NULL_ITEM;
}

template <class KeyType, class ObjType>
void
JHashtable<KeyType,ObjType>::clear()
{
	JHashtableEntry** tab = m_table;
	
	for(UINT i = 0; i < m_tableSize; i++ )
	{
		JHashtableEntry* e = tab[i];
		while( e != NULL )
		{
			JHashtableEntry* tmp = e;
			e = e->next;
			delete tmp;
		}
		
		tab[i] = NULL;
	}

	//reset the counter
	m_count = 0;
}

template <class KeyType, class ObjType>
UINT
JHashtable<KeyType,ObjType>::nextPrime( UINT start )
{
	if( start % 2 == 0 )
		start++;

	UINT i;

	for( ; ; start += 2 )
	{
		for( i = 3; i * i <= start; i += 2 )
			if( start % i == 0 )
				break;

		if( i * i > start )
			return start;
	}
}

template <class KeyType, class ObjType>
JVector<KeyType> 
JHashtable<KeyType,ObjType>::keys()
{
	JVector<KeyType> results;

	JHashtableEntry** tab = m_table;
	
	for(UINT i = 0; i < m_tableSize; i++ )
	{
		JHashtableEntry* e = tab[i];
		while( e != NULL )
		{
			results.addElement(e->key);
			e = e->next;
		}
	}

	return results;
}



