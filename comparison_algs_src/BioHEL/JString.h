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
   Visit http://www.mike95.com/c_plusplus/classes/JString/
 
  This library avoids the need to use pointers to char or manual
   memory manipulation while working with Strings.  This class has been
   based on the Java String class, and thus has all of the Java public functionality
   with a few extras [replace()] needed for useful functionality.
 
  //The following people have contributed to the solution
   //of bugs or improvements in this library
   //=====================================================
   //Carl Pupa, [pumacat@erols.com]
   //Thomas Watson, [w@tson.dk]
   //Subbiah, Venkat [vsubbiah@corvis.com]
   //Oren Tirosh [mailto:oren@hishome.net] 
*/ 

 #ifndef __JString
 #define __JString
 
 #include <iostream>
 #include <cstring>
 #include <stdlib.h>
 #include "M95_types.h"
 
using namespace std;

 class JString {
 public:
     // Constructor(s) / Destructors
     //-----------------------------
     JString( const char *Value = "" );
     JString( const JString &Value );
     virtual ~JString() { delete [] Buffer; }
 
    //Operators
     //----------
     const JString &    operator = ( const JString &Rhs );
     const JString &    operator +=( const JString &Rhs );
     const JString &    operator +=( const char );
     int                operator ==( const JString &Rhs ) const;
     int                operator !=( const JString &Rhs ) const;
     int                operator < ( const JString &Rhs ) const;
     int                operator > ( const JString &Rhs ) const;
     int                operator <=( const JString &Rhs ) const;
     int                operator >=( const JString &Rhs ) const;
     char            operator []( UINT Index ) const;
     char&            operator []( UINT Index );
 
    //Methods
     //-------
     //INSPECTORS
     //==========
     char            charAt( UINT index ) const;
     int                compareTo( const JString &anotherString ) const;
     const char*        cstr( ) const { return Buffer; }
     int                endsWith( const JString &suffix ) const;
     int                equals( const JString &anObject ) const;
     int                equalsIgnoreCase( const JString &anotherString ) const;
     int                indexOf( char ch ) const;
     int                indexOf( char ch, UINT fromIndex ) const;
     int                indexOf( const JString &str ) const;
     int                indexOf( const JString &str, UINT fromIndex ) const;
     int                lastIndexOf( char ch ) const;
     int                lastIndexOf( char ch, UINT fromIndex ) const;
     int                lastIndexOf( const JString &str ) const;
     int                lastIndexOf( const JString &str, UINT fromIndex ) const;
     UINT        length( ) const { return Length; }
     int                startsWith( const JString &prefix ) const;
     int                startsWith( const JString &prefix, UINT toffset ) const;
     JString            substring( UINT beginIndex ) const;
     JString            substring( UINT beginIndex, UINT endIndex ) const;
     JString            toLowerCase( ) const;
     JString            toUpperCase( ) const;
     JString            trim( ) const; 

     //Methods
     //-------
     //MODIFIERS
     //=========
     const JString&    concat( const JString &str );
     void            replace( char oldChar, char newChar );
     void            replace( const JString& match, const JString& replace );
     void            Format( const char* fmt, ...);
 
    // Friends
     //--------
     friend JString        operator + ( JString Lhs, const JString &Rhs );
     friend ostream&        operator<< ( ostream &Out, const JString &Value );
     friend istream&        operator>> ( istream &In, JString &Value );
 
protected:
     //Members
     //-------
     char *Buffer;               // Stores the chars
     UINT BufferLen;              // Max strlen for Buffer
     UINT Length;                 // Length of string
 
    void GetBuffer(UINT MaxStrLen);
     void Double( );
     void verifyIndex( UINT number ) const;
 };
 
//Class Functions
 //===============
 inline void
 JString::GetBuffer(UINT MaxStrLen)
 {
     BufferLen = MaxStrLen;
     Buffer = new char[BufferLen + 1];
 }
 
inline void
 JString::Double( )
 {
     char *temp = Buffer;
     GetBuffer( ++BufferLen * 2 );
     strcpy( Buffer, temp );
     delete [] temp;
 }
 
inline void
 JString::verifyIndex( UINT index ) const
 {
     if ( index >= Length )
     {
         //throw "Index Out Of Bounds Exception";
         cerr << "Index Out Of Bounds Exception at [" 
            << index << "] in:\n" << Buffer << endl;
         exit(1);
     }
 }
 
//Friend Functions
 //================
 inline JString
 operator+( JString Lhs, const JString &Rhs ) 
{
     return Lhs += Rhs;
 }
 
#endif
 
 



 
