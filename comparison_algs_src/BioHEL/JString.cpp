 /*******************************************
   Downloaded from: http://www.mike95.com
   Copyright (c)1997 Michael Olivero
   All Rights Reserved
 ********************************************/
 #include "JString.h"
 #include <ctype.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <stdarg.h>
 
//============
 //Constructors
 //============
 JString::JString( const char *Value )
 {
     if ( Value == NULL )
         Value = "";
 
    GetBuffer( Length = strlen( Value ) );
     strcpy( Buffer, Value );
 }
 
JString::JString( const JString &Value )
 {
     GetBuffer( Length = Value.Length );
     strcpy( Buffer, Value.Buffer );
 }
 
char
 JString::charAt( UINT loc ) const
 {
     return operator[]( loc );
 }
 
int
 JString::compareTo( const JString &s2 ) const
 {
     return strcmp( Buffer, s2.Buffer );
 }
 
const JString &
 JString::concat( const JString &s2 )
 {
     return (*this) += s2;
 }
 
const JString &
 JString::operator=( const JString &Rhs )
 {
     if ( this == &Rhs )
         return *this;
 
    if ( Rhs.Length > Length )
     {
         delete [] Buffer;
         GetBuffer( Rhs.Length );
     }
 
    Length = Rhs.Length;
     strcpy( Buffer, Rhs.Buffer );
 
    return *this;
 }
 
const JString &
 JString::operator+=( const char aChar )
 {
     if ( Length == BufferLen )
         Double();
 
    Buffer[ Length++ ] = aChar;
     Buffer[ Length ] = '\0';
 
    return *this;
 }
 
const JString &
 JString::operator+=( const JString &other )
 {
     Length += other.Length;
     if ( Length > BufferLen )
     {
         char *temp = Buffer;
         GetBuffer( Length );
         strcpy( Buffer, temp );
         delete [] temp;
     }
     strcat( Buffer, other.Buffer );
 
    return *this;
 }
 

 int
 JString::operator==( const JString &Rhs ) const
 {
     return ( Length == Rhs.Length && strcmp( Buffer, Rhs.Buffer ) == 0 );
 }
 
int
 JString::operator!=( const JString &Rhs ) const
 {
     return ( Length != Rhs.length() || strcmp( Buffer, Rhs.cstr() ) != 0 );
 }
 
int
 JString::operator<( const JString &Rhs ) const
 {
     return strcmp( Buffer, Rhs.Buffer ) < 0;
 }
 
int
 JString::operator>( const JString &Rhs ) const
 {
     return strcmp( Buffer, Rhs.Buffer ) > 0;
 }
 
int
 JString::operator<=( const JString &Rhs ) const
 {
     return strcmp( Buffer, Rhs.Buffer ) <= 0;
 }
 
int
 JString::operator>=( const JString & Rhs ) const
 {
     return strcmp( Buffer, Rhs.Buffer ) >= 0;
 }
 
char &
 JString::operator[]( UINT Index )
 {
     verifyIndex( Index );
     return Buffer[ Index ];
 }
 
char
 JString::operator[]( UINT Index ) const
 {
     verifyIndex( Index );
     return Buffer[ Index ];
 }
 

 int
 JString::endsWith( const JString &s2 ) const
 {
     if ( Length < s2.Length )
         return 0;
 
    return strcmp( &Buffer[ Length - s2.Length], s2.cstr() ) == 0;
 }
 
int
 JString::equals( const JString &s2 ) const
 {
     return ( Length == s2.Length && strcmp( Buffer,s2.Buffer ) == 0 );
 }
 

 int
 JString::equalsIgnoreCase( const JString &s2 ) const
 {
     if ( this == &s2 )
         return 1;
     else if ( Length != s2.Length )
         return 0;
 
    return strcmp(toLowerCase().cstr(), s2.toLowerCase().cstr()) == 0;
 }
 
void
 JString::Format( const char* fmt, ... )
 {
     va_list iterator;
     va_start( iterator, fmt );
 

     va_end( iterator );
 }
 

 void
 JString::replace( char findChar, char replaceChar )
 {
     char* temp = Buffer;
     while( temp = strchr( temp, findChar ) )
         *temp = replaceChar;
 }
 void
 JString::replace( const JString& match, const JString& replace )
 {
     JString temp = Buffer, newString;
 
    int loc;
     while ( (loc = temp.indexOf( match )) != -1 )
     {
         newString += temp.substring( 0, loc );
         newString += replace;
         temp = temp.substring( loc + match.Length );
     } 

     newString += temp;                   //get rest of string;
 
    *this = newString;
 }
 
int
 JString::indexOf( char temp ) const
 {
     return indexOf( temp, 0 );
 }
 
int
 JString::indexOf( char ch, UINT fromIndex ) const
 {
     if ( fromIndex >= Length )
         return -1;
 
    const char* temp = strchr( &Buffer[fromIndex], ch );
     if ( temp == NULL )
         return -1;
 
    return temp - Buffer;
 }
 
int
 JString::indexOf( const JString &s2 ) const
 {
     return indexOf( s2, 0 );
 }
 

 int
 JString::indexOf( const JString &s2, UINT fromIndex ) const
 {
     if ( fromIndex >= Length )
         return -1;
 
    const char *theFind = strstr( &Buffer[ fromIndex ], s2.cstr() );
 
    if ( theFind == NULL )
         return -1;
 
    return theFind - Buffer; //pointer substraction
 }
 

 
int
 JString::lastIndexOf( char theChar ) const
 {
     return lastIndexOf( theChar, Length - 1 );
 }
 
int
 JString::lastIndexOf( char ch, UINT fromIndex ) const
 {
     if ( fromIndex >= Length )
         return -1;
 
    char tempchar = Buffer[fromIndex + 1];
     Buffer[fromIndex + 1] = '\0';
     char* temp = strrchr( Buffer, ch );
     Buffer[fromIndex + 1] = tempchar;
 
    if ( temp == NULL )
         return -1;
 
    return temp - Buffer;
 }
 
int
 JString::lastIndexOf( const JString &s2 ) const
 {
     return lastIndexOf( s2, Length - s2.Length );
 }
 
int
 JString::lastIndexOf( const JString &s2, UINT fromIndex ) const
 {
     //=============================
     //avoid check for empty strings
     //=============================
     if ( s2.Length == 0 || s2.Length - 1 > fromIndex || 
        fromIndex >= Length )
     return -1;
 
    //========================
     //matching first character
     //========================
     char temp = s2[ 0 ];
 
    for ( int i = fromIndex; i >= 0; i-- )
     {
         if ( Buffer[ i ] == temp &&
             (*this).substring( i, i + s2.Length ).equals( s2 ) )
             return i;
     }
     return -1;
 }
 
int
 JString::startsWith( const JString &s2 ) const
 {
     if ( Length < s2.Length )
         return 0;
 
    return startsWith( s2, 0 );
 }
 
int
 JString::startsWith( const JString &s2, UINT offset ) const
 {
     if ( offset > Length - s2.Length )
         return 0;
 
    return strncmp( &Buffer[offset], s2.cstr(), s2.Length ) == 0;
 }
 
JString
 JString::substring( UINT left ) const
 {
     return substring( left, Length );
 }
 
JString
 JString::substring( UINT left, UINT right ) const
 {
     if ( left > right )
     {
         int temp = right;
         right = left;
         left = temp;
     }
 
    if ( right > Length )
     {
         cerr << "Index Out Of Bounds Exception w/ substring(" << left 
            << "," << right << "]:\n" << Buffer << endl;
         exit(1);
     }
 
    char temp = Buffer[ right ]; //save the replaced character
     Buffer[ right ] = '\0';      //nullify the the character
 
    JString outPut = ( Buffer + left ); //Pointer arithmetic
 
    Buffer[ right ] = temp;      //restore character
 
    return outPut;
 }
 
JString
 JString::toLowerCase( ) const
 {
     JString temp = Buffer;
 
    for ( UINT i = 0; i < Length; i++ )
         temp.Buffer[ i ] = tolower( temp.Buffer[ i ] );
 
    return temp;
 }
 
JString
 JString::toUpperCase() const
 {
     JString temp = Buffer;
 
    for ( UINT i = 0; i < Length; i++ )
         temp.Buffer[ i ] = toupper( temp.Buffer[ i ] );
 
    return temp;
 }
 
JString
 JString::trim() const
 {
     JString temp = Buffer;
     UINT i,j;
 
    for ( i = 0; i < Length; i++ )
     {
         if ( !isspace(Buffer[i]) )
             break;
     }
 
    for ( j = temp.Length - 1; j > i; j-- )
     {
         if ( !isspace(Buffer[j]) )
             break;
     }
 
    return temp.substring( i, j + 1);
 }
 
istream &
 operator >> ( istream &In, JString &Value )
 {
     static char Str[ 2048 ]; // allocate max size of 2048 characters
 
    In >> Str;
 
    Value = Str;             // assign to reference JString
 
    return In;               // Return istream
 }
 
ostream &
 operator << ( ostream &Out, const JString &Value )
 {
     Out << Value.Buffer;
     return Out;
 }
 
 



 
