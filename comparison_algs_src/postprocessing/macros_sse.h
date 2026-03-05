#ifndef _MACROS_SSE_
#define _MACROS_SSE_

/*typedef int __v2di __attribute__ ((mode (V2DI),aligned(8)));
typedef int __v4si __attribute__ ((mode (V4SI),aligned(8)));
typedef int __v16qi __attribute__ ((mode (V16QI),aligned(8)));
#define __m128i __v2di

typedef float __v4sf __attribute__ ((__mode__(__V4SF__)));
typedef float __m128 __attribute__ ((__mode__(__V4SF__)));

#define _MM_SHUFFLE(fp3,fp2,fp1,fp0) \
 (((fp3) << 6) | ((fp2) << 4) | ((fp1) << 2) | (fp0))

static __inline __m128
_mm_load_ps (float const *__P)
{
  return (__m128) __builtin_ia32_loadaps (__P);
}

static __inline __m128
_mm_set1_ps (float __F)
{
  __v4sf __tmp = __builtin_ia32_loadss (&__F);
  return (__m128) __builtin_ia32_shufps (__tmp, __tmp, _MM_SHUFFLE (0,0,0,0));
}

static __inline __m128
_mm_set_ps1 (float __F)
{
  return _mm_set1_ps (__F);
}

static __inline __m128
_mm_sub_ps (__m128 __A, __m128 __B)
{
  return (__m128) __builtin_ia32_subps ((__v4sf)__A, (__v4sf)__B);
}

static __inline void
_mm_store_ps (float *__P, __m128 __A)
{
  __builtin_ia32_storeaps (__P, (__v4sf)__A);
}



static __inline __m128
_mm_cmpgt_ps (__m128 __A, __m128 __B)
{
  return (__m128) __builtin_ia32_cmpgtps ((__v4sf)__A, (__v4sf)__B);
}

static __inline __m128i
_mm_or_si128 (__m128i __A, __m128i __B)
{
  return (__m128i)__builtin_ia32_por128 ((__v2di)__A, (__v2di)__B);
}

static __inline __m128
_mm_andnot_ps (__m128 __A, __m128 __B)
{
  return __builtin_ia32_andnps (__A, __B);
}

static __inline __m128i
_mm_andnot_si128 (__m128i __A, __m128i __B)
{
  return (__m128i)__builtin_ia32_pandn128 ((__v2di)__A, (__v2di)__B);
}

static __inline __m128i
_mm_and_si128 (__m128i __A, __m128i __B)
{
  return (__m128i)__builtin_ia32_pand128 ((__v2di)__A, (__v2di)__B);
}

static __inline int
_mm_movemask_epi8 (__m128i __A)
{
  return __builtin_ia32_pmovmskb128 ((__v16qi)__A);
}

static __inline int
_mm_movemask_ps (__m128 __A)
{
  return __builtin_ia32_movmskps ((__v4sf)__A);
}*/

#define VEC_MATCH(vecFLB,fLB,vecFUB,fUB,vecINS,fIN,vecTmp,vecOne,vecRes) {\
        vecFLB = _mm_load_ps(fLB);\
        vecFUB = _mm_load_ps(fUB);\
        vecINS = _mm_load_ps(fIN);\
        \
        vecRes = (__m128i)_mm_cmpgt_ps(vecFUB,vecFLB);\
        vecTmp = _mm_or_si128(\
                (__m128i)_mm_cmpgt_ps(vecFLB,vecINS),\
                (__m128i)_mm_cmpgt_ps(vecINS,vecFUB)\
                );\
        vecRes = _mm_andnot_si128(_mm_and_si128(vecRes,vecTmp),vecOne);\
}

#define VEC_MATCH2(vecRule,Rule,vecIns,Ins,vecABS,vecRes) {\
        vecRule = _mm_load_ps(Rule);\
        vecIns = _mm_load_ps(Ins);\
        \
        vecIns = _mm_andnot_ps(vecABS,vecIns);\
        vecRes = _mm_cmpgt_ps(vecIns,vecRule);\
}



#define VEC_TRANS(vRule,rule,vIns,ins) {\
        vRule = _mm_load_ps(rule);\
        vIns = _mm_load_ps(ins);\
        vIns = _mm_sub_ps(vIns,vRule);\
        _mm_store_ps(ins,vIns);\
}



#endif
