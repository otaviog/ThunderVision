grammar ThunderLang;

options
{
	language = C;
}

@parser::postinclude
{
#include <antlr3.h>
#include <stdlib.h>
#include "tlctx.h"

static char* RemoveQuotMark(ANTLR3_STRING *quotedString)
{
	if ( quotedString->len > 0 )
	{
		quotedString->chars[quotedString->len - 1] = 0;
	}
	
	return quotedString->chars + 1;
}

#define MY_TXT(token) token->getText(token)
#define MY_TXT_C(token) MY_TXT(token)->chars
#define MY_TXT_SC(token) RemoveQuotMark(MY_TXT(token))
#define MY_INT(token) atoi(MY_TXT_C(token))
#define MY_FLT(token) (float)(atof(MY_TXT_C(token)))
#define MY_DBL(token) (atof(MY_TXT_C(token)))
}

ID  :	('a'..'z'|'A'..'Z'|'_') ('a'..'z'|'A'..'Z'|'0'..'9'|'_')*
    ;

//INT :	'0'..'9'+
  //	  ;

FLOAT
    :  ('+'|'-')? (
    	('0'..'9')+ ('.' ('0'..'9')* )? 
      	| '.' ('0'..'9')+ 
      	) EXPONENT?
    ;

COMMENT
    :   '//' ~('\n'|'\r')* '\r'? '\n' {$channel=HIDDEN;}
    |   '/*' ( options {greedy=false;} : . )* '*/' {$channel=HIDDEN;}
    ;

WS  :   ( ' '
        | '\t'
        | '\r'
        | '\n'
        ) {$channel=HIDDEN;}
    ;

STRING
    :  '"' ( ESC_SEQ | ~('\\'|'"') )* '"'
    ;

CHAR:  '\'' ( ESC_SEQ | ~('\''|'\\') ) '\''
    ;

fragment
EXPONENT : ('e'|'E') ('+'|'-')? ('0'..'9')+ ;

fragment
HEX_DIGIT : ('0'..'9'|'a'..'f'|'A'..'F') ;

fragment
ESC_SEQ
    :   '\\' ('b'|'t'|'n'|'f'|'r'|'\"'|'\''|'\\')
    |   UNICODE_ESC
    |   OCTAL_ESC
    ;

fragment
OCTAL_ESC
    :   '\\' ('0'..'3') ('0'..'7') ('0'..'7')
    |   '\\' ('0'..'7') ('0'..'7')
    |   '\\' ('0'..'7')
    ;

fragment
UNICODE_ESC
    :   '\\' 'u' HEX_DIGIT HEX_DIGIT HEX_DIGIT HEX_DIGIT
    ;


thunderLang [void *ctxobj]
	:	(camerasDesc[ctxobj] | stereo[ctxobj]) * ;

camerasInput [void *ctxobj]
	:	'CamerasInput' id=ID '=' 'dev' INT ',' 'dev' INT
	;
	
camerasDesc [void *ctxobj]
	: 'CamerasDesc' id=ID '{' camerasDescOpts[ctxobj, MY_TXT_C(id)]* '}' 
	;
	
camerasDescOpts [void *ctxobj, const char *descId]
	@init { double mtx[9]; }
	: cameraParms[ctxobj, descId]
	| 'fundamental' '=' matrix33[mtx] { tlcSetFundamental(ctxobj, descId, mtx); }
	| 'extrinsics' '=' '[' matrix33[mtx] '[' t1=FLOAT t2=FLOAT t3=FLOAT ']' ']' { tlcSetExtrinsic(ctxobj, descId, mtx, MY_DBL(t1), MY_DBL(t2), MY_DBL(t3)); }
	;
	
cameraParms [void *ctxobj, const char *descId]
	@init { int leftOrRight = 0; }
	:	'Camera' ('Left' { leftOrRight = 0; } | 'Right' { leftOrRight = 1; } ) '{' cameraParmsOpts[ctxobj, descId, leftOrRight]* '}'
	;

cameraParmsOpts[void *ctxobj, const char *descId, int leftOrRight]
	@init { double mtx[9]; }
	: 'intrinsic_transform' '=' matrix33[mtx] { tlcSetIntrinsic(ctxobj, descId, leftOrRight, mtx); } 
	| 'intrinsic_distortion' '=' '[' d1=FLOAT d2=FLOAT d3=FLOAT d4=FLOAT d5=FLOAT ']' 
		{ tlcSetDistortion(ctxobj, descId, leftOrRight, MY_DBL(d1), MY_DBL(d2), MY_DBL(d3), MY_DBL(d4), MY_DBL(d5)); }	
	;

matrix33 [ double *mtx ]
	: '[' 
	'[' m11=FLOAT m12=FLOAT m13=FLOAT ']' 
	'[' m21=FLOAT m22=FLOAT m23=FLOAT ']'
       	'[' m31=FLOAT m32=FLOAT m33=FLOAT ']' 
       	']'
       	 {
       	  mtx[0] = MY_DBL(m11); mtx[1] = MY_DBL(m12); mtx[2] = MY_DBL(m13); 
       	  mtx[3] = MY_DBL(m21); mtx[4] = MY_DBL(m22); mtx[5] = MY_DBL(m23); 
       	  mtx[6] = MY_DBL(m31); mtx[7] = MY_DBL(m32); mtx[8] = MY_DBL(m33); 
       	 }
	;

stereo [void *ctxobj]
	: 'Stereo' '{' '}'
	;




